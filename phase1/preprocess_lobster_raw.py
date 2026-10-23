"""Preprocess raw LOBSTER archives into compact per-stock-day files (event level).

Streams each needed day out of its 7z archive (nothing is extracted to disk), keeps the
top-of-book events in the trading window, and writes <out>/<TICKER>/<YYYY-MM-DD>.npz plus
a small .json with counts. Finished days are skipped, so an interrupted run resumes.

Example
-------
    python preprocess_lobster_raw.py --months 2019-10 2020-03 --n-tickers 10 --dry-run
    python preprocess_lobster_raw.py --months 2019-10 2020-03 --n-tickers 10 --workers 16

Tickers: by default the n with the largest archives (a proxy for message activity) among
those whose archives cover every requested month, so both months use the same stocks.
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from phase1 import lobster_raw as R


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def plan(args):
    archives = R.find_archives(args.raw_root)
    log(f"found {len(archives)} ticker archives under {args.raw_root}")
    tickers, per_month = R.select_tickers(archives, args.months, args.n_tickers, args.tickers)
    log(f"tickers: {', '.join(tickers)}")
    jobs = []
    for m in args.months:
        for t in tickers:
            arc = per_month[m][t]
            days = {d: f for d, f in R.list_days(arc).items() if d.startswith(m)}
            log(f"  {t} {m}: {len(days)} trading days in {arc.path.name} ({arc.size / 1e9:.1f} GB)")
            for d, (msg, book) in days.items():
                jobs.append(dict(archive=str(arc.path), message=msg, orderbook=book, levels=arc.levels,
                                 out=str(Path(args.out) / t / f"{d}.npz"),
                                 trim_open_s=args.trim_open_s, trim_close_s=args.trim_close_s))
    return tickers, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/nfs/data/lobster_data/lobster_raw")
    ap.add_argument("--out", default="data/lobster_events")
    ap.add_argument("--months", nargs="+", default=["2019-10", "2020-03"])
    ap.add_argument("--n-tickers", type=int, default=10)
    ap.add_argument("--tickers", nargs="+", default=None, help="explicit tickers instead of the largest archives")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--trim-open-s", type=float, default=300.0)
    ap.add_argument("--trim-close-s", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="only show the plan")
    args = ap.parse_args()

    tickers, jobs = plan(args)
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    log(f"{len(jobs)} stock-days planned, {len(jobs) - len(todo)} already done, {len(todo)} to process")
    if args.dry_run or not todo:
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "preprocess_config.json").write_text(json.dumps(dict(vars(args), tickers_selected=tickers), indent=4))
    t0 = time.perf_counter()
    errors = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(R.process_day, todo), 1):
            el = time.perf_counter() - t0
            name = "/".join(Path(res["out"]).parts[-2:])
            if res["status"] == "ok":
                log(f"{k}/{len(todo)} {name}: {res['n_messages']:,} messages, {res['n_kept']:,} top-of-book events "
                    f"({res['n_invalid']:,} invalid) | elapsed {el / 60:.1f} min, ETA {el / k * (len(todo) - k) / 60:.1f} min")
            elif res["status"] == "error":
                errors.append(res)
                log(f"{k}/{len(todo)} {name}: ERROR {res['error']}")
    log(f"done in {(time.perf_counter() - t0) / 60:.1f} min; {len(errors)} error(s)")
    if errors:
        (out / "preprocess_errors.json").write_text(json.dumps(errors, indent=4))


if __name__ == "__main__":
    main()
