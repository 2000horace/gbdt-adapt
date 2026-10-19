"""Fast layer of the proposal (Sec. 3.1-3.4): robust, lazily discounted leaf filter.

Per stream event i:
    1. the label of event j = i - h becomes observable -> update the filter with it;
    2. predict event i from the current state.

Implementation notes (deviations from proposal v2, to fold back into the text):
    * The update uses the CURRENT leaf means to form the residual, not the stored
      prediction from event j. This is the correct Kalman update given the state.
    * The Student-t scale is updated with the EM fixed point
      sigma2 = sum(w * omega * r^2) / sum(w), i.e. dividing by the discounted count,
      not by the discounted sum of omegas. Dividing by sum(omega) biases it upward.
    * Lazy discounting uses a cumulative "discount clock" so it stays exact when
      delta changes during an alarm window.
    * The detector statistic is u = 1 - min(omega, 1), which lies in [0, 1).
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class AdaptiveConfig:
    # Leaf filter.
    P0: float = 1e-4              # prior variance of each tree leaf
    P0_bias: float = 1e-3         # prior variance of the bias leaf
    P_max_factor: float = 4.0     # cap on leaf variance, as a multiple of its prior
    delta: float = 0.999          # discount per event
    nu: float = 4.0               # Student-t degrees of freedom
    overlap_factor: float = None  # noise inflation for overlapping labels; None -> h
    # Noise scale.
    sigma2_init: float = 1.0
    delta_v: float = 0.995
    sigma2_min: float = 1e-4
    # Calibration factor.
    alpha: float = 0.2            # target miss rate of the central interval
    eta_c: float = 0.005
    # Detector.
    detector: str = "bounded"     # "bounded" (u = 1 - min(omega, 1)) or "raw" (u = z^2)
    zeta: float = 0.05
    lam: float = 30.0
    u_bar_rate: float = 0.001     # EWMA rate of the detector baseline
    warmup: int = 2_000           # updates before the detector is armed
    alarm_window: int = 2_000
    delta_alarm: float = 0.99
    nu_alarm: float = 30.0
    # Ablation switches.
    use_robust: bool = True
    use_discount: bool = True
    use_scale: bool = True
    use_calibration: bool = True
    use_detector: bool = True
    use_alarm_action: bool = True


class AdaptiveLeafFilter:
    def __init__(self, w0, bias_index, cfg: AdaptiveConfig, h: int):
        self.cfg = cfg
        self.h = h
        self.m = w0.astype(float).copy()
        self.P0 = np.full(len(w0), cfg.P0)
        self.P0[bias_index] = cfg.P0_bias
        self.P = self.P0.copy()
        self.P_max = cfg.P_max_factor * self.P0
        self.clock_at_touch = np.zeros(len(w0))   # discount clock at last touch
        self.clock = 0.0                          # cumulative -log(delta)
        self.ofac = float(h if cfg.overlap_factor is None else cfg.overlap_factor)

        self.sigma2 = cfg.sigma2_init
        self.n_eff = 1.0
        self.d_eff = cfg.sigma2_init
        self.c = 1.0
        self.q = stats.t.ppf(1.0 - cfg.alpha / 2.0, cfg.nu)

        self.S = 0.0
        self.u_bar = None
        self.n_updates = 0
        self.alarm_until = -1
        self.alarms = []

        # Diagnostics recorded at every update.
        self.log_r = []
        self.log_dpred = []
        self.log_omega = []

    # ------------------------------------------------------------------ helpers
    def _in_alarm(self, now):
        return self.cfg.use_alarm_action and now <= self.alarm_until

    def _tick(self, now):
        """Advance the discount clock by one event."""
        if not self.cfg.use_discount:
            return
        delta = self.cfg.delta_alarm if self._in_alarm(now) else self.cfg.delta
        self.clock += -np.log(delta)

    def _touch(self, idx):
        """Apply pending discount to the given leaves and return their variances."""
        growth = np.exp(np.minimum(self.clock - self.clock_at_touch[idx], 700.0))
        P = np.minimum(self.P[idx] * growth, self.P_max[idx])
        self.P[idx] = P
        self.clock_at_touch[idx] = self.clock
        return P

    # ------------------------------------------------------------------ API
    def predict(self, idx):
        """Return (mean, t-scale^2, df) of the predictive distribution."""
        P = self._touch(idx)
        mu = self.m[idx].sum()
        s = self.c * (P.sum() + self.sigma2)
        return mu, s, self.cfg.nu

    def update(self, idx, y, now):
        cfg = self.cfg
        P = self._touch(idx)
        S0 = P.sum()
        mu = self.m[idx].sum()
        s = self.c * (S0 + self.sigma2)
        r = y - mu
        z2 = r * r / s

        nu_upd = cfg.nu_alarm if self._in_alarm(now) else cfg.nu
        omega = (nu_upd + 1.0) / (nu_upd + z2) if cfg.use_robust else 1.0

        # Leaf update with noise inflated for outliers and for label overlap.
        s_tilde = S0 + self.ofac * self.sigma2 / omega
        gain = P / s_tilde
        self.m[idx] += gain * r
        self.P[idx] = P - P * P / s_tilde

        # Noise scale: discounted EM fixed point for the Student-t scale.
        if cfg.use_scale:
            dv = cfg.delta_v
            self.n_eff = dv * self.n_eff + 1.0
            # Do NOT subtract S0 here: the diagonal sum of leaf variances overstates
            # the variance of the prediction (co-active leaves become negatively
            # correlated through the updates), so r^2 - S0 is often negative and
            # drives sigma2 to the floor, which makes the gains explode. Any
            # double counting of S0 in s is absorbed by the calibration factor c.
            self.d_eff = dv * self.d_eff + omega * r * r
            self.sigma2 = max(self.d_eff / self.n_eff, cfg.sigma2_min)

        # Calibration: widen after misses, shrink after hits.
        if cfg.use_calibration:
            miss = float(abs(r) > self.q * np.sqrt(s))
            self.c *= np.exp(cfg.eta_c * (miss - cfg.alpha))

        # Drift detector.
        self.n_updates += 1
        if cfg.use_detector:
            u = (1.0 - min(omega, 1.0)) if cfg.detector == "bounded" else z2
            if self.u_bar is None:
                self.u_bar = u
            if self.n_updates <= cfg.warmup:
                self.u_bar += (u - self.u_bar) / self.n_updates
            else:
                if not self._in_alarm(now):
                    self.u_bar += cfg.u_bar_rate * (u - self.u_bar)
                self.S = max(0.0, self.S + u - self.u_bar - cfg.zeta)
                if self.S > cfg.lam:
                    self.alarms.append(now)
                    self.alarm_until = now + cfg.alarm_window
                    self.S = 0.0

        self.log_r.append(r)
        self.log_dpred.append(gain.sum() * r)   # change of the prediction at this x
        self.log_omega.append(omega)

    def run(self, idx_stream, y_obs):
        """Prequential loop over the stream. Returns predictive (mu, s, df) arrays."""
        T = len(idx_stream)
        mu = np.empty(T)
        s = np.empty(T)
        for i in range(T):
            self._tick(i)
            j = i - self.h
            if j >= 0:
                self.update(idx_stream[j], y_obs[j], now=i)
            mu[i], s[i], _ = self.predict(idx_stream[i])
        return mu, s, np.full(T, self.cfg.nu)
