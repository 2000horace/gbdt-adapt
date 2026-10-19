"""Pre-trained boosted ensemble and its leaf-indicator feature map.

With frozen structure, F(x) = sum_k w[k, leaf_k(x)] + base. We give every leaf
a global index and add one extra always-active "bias leaf" holding the base
score, so F(x) = sum over the K + 1 active indices of w[idx].
"""

from dataclasses import dataclass

import numpy as np
import xgboost as xgb


@dataclass
class EnsembleConfig:
    n_trees: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    reg_lambda: float = 1.0
    subsample: float = 0.8
    min_child_weight: float = 20.0
    seed: int = 0
    device: str = "cpu"           # "cuda" to train with XGBoost on the GPU

    def params(self):
        return {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "lambda": self.reg_lambda,
            "subsample": self.subsample,
            "min_child_weight": self.min_child_weight,
            "seed": self.seed,
            "nthread": 1,
            "device": self.device,
            "verbosity": 0,
        }


def train_ensemble(X, y, cfg: EnsembleConfig):
    dtrain = xgb.DMatrix(X, label=y)
    return xgb.train(cfg.params(), dtrain, num_boost_round=cfg.n_trees)


class LeafMap:
    """Maps inputs to the global indices of their active leaves."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster
        df = booster.trees_to_dataframe()
        leaves = df[df["Feature"] == "Leaf"]
        self.n_trees = int(df["Tree"].max()) + 1

        # Per-tree lookup table: node id -> global leaf index.
        self._lookup = []
        values = []
        offset = 0
        for k in range(self.n_trees):
            lk = leaves[leaves["Tree"] == k]
            table = np.full(int(df[df["Tree"] == k]["Node"].max()) + 1, -1, dtype=np.int64)
            table[lk["Node"].to_numpy()] = offset + np.arange(len(lk))
            self._lookup.append(table)
            values.append(lk["Gain"].to_numpy())  # "Gain" holds the leaf value for leaves
            offset += len(lk)

        self.n_leaves = offset                    # D, excluding the bias leaf
        self.bias_index = offset                  # index of the bias leaf
        self.size = offset + 1                    # D + 1
        w = np.concatenate(values)
        self.w0 = np.append(w, 0.0)               # bias value set by calibrate_bias
        self._bias_set = False

    def leaf_index(self, X):
        """Global active indices, shape (n, K + 1); last column is the bias leaf."""
        raw = self.booster.predict(xgb.DMatrix(X), pred_leaf=True).astype(np.int64)
        out = np.empty((len(X), self.n_trees + 1), dtype=np.int64)
        for k in range(self.n_trees):
            out[:, k] = self._lookup[k][raw[:, k]]
        out[:, -1] = self.bias_index
        if (out < 0).any():
            raise RuntimeError("leaf lookup failed")
        return out

    def calibrate_bias(self, X):
        """Set the bias leaf so that w0[idx].sum(1) reproduces booster.predict."""
        idx = self.leaf_index(X)
        pred = self.booster.predict(xgb.DMatrix(X))
        gap = pred - self.w0[idx[:, :-1]].sum(axis=1)
        if gap.std() > 1e-4:
            raise RuntimeError(f"base score is not constant (std {gap.std():.2e})")
        self.w0[self.bias_index] = gap.mean()
        self._bias_set = True
        return idx

    def predict(self, idx, w=None):
        w = self.w0 if w is None else w
        return w[idx].sum(axis=1)
