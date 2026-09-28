"""Paired OOF audit for one fixed supervised Uni-Mol2 mode.

No mode is chosen with outer-test results. Use this only after the candidate
run has completed all five frozen outer scaffold folds for both tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("hepg2_pIC50", "hct116_pIC50")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", required=True)
    ap.add_argument("--candidate-model", required=True)
    ap.add_argument("--matched-2d-dir", required=True)
    ap.add_argument("--full-2d-file", default="experiments/optimization_v1/nested_cv/outer_oof_predictions.csv")
    ap.add_argument("--full-2d-model", default="2d_best")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()

    croot, mroot = Path(args.candidate_dir), Path(args.matched_2d_dir)
    fpath, out = Path(args.full_2d_file), Path(args.output_dir)
    if not croot.is_absolute(): croot = ROOT / croot
    if not mroot.is_absolute(): mroot = ROOT / mroot
    if not fpath.is_absolute(): fpath = ROOT / fpath
    if not out.is_absolute(): out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    cpath = croot / "outer_oof_predictions.csv"
    mpath = mroot / "outer_oof_predictions.csv"
    candidate = pd.read_csv(cpath)
    matched = pd.read_csv(mpath)
    full = pd.read_csv(fpath)
    full = full.loc[full.model.astype(str) == args.full_2d_model].copy()
    candidate = candidate.loc[candidate.model.astype(str) == args.candidate_model].copy()
    if candidate.empty or full.empty:
        raise RuntimeError("Candidate or full-training 2D predictions are empty")

    summaries, fold_rows, intervals = [], [], []
    paired_rows = []
    for ti, task in enumerate(TASKS):
        c = candidate.loc[candidate.task == task].copy()
        if set(c.outer_fold.astype(int).unique()) != set(range(5)):
            raise RuntimeError(f"Expected all five candidate folds for {task}")
        if c.smiles.duplicated().any():
            raise RuntimeError(f"Duplicate candidate OOF molecule for {task}")
        for base_name, base_all in (("matched_2d", matched), ("full_training_2d", full)):
            b = base_all.loc[base_all.task == task].copy()
            if b.smiles.duplicated().any():
                raise RuntimeError(f"Duplicate {base_name} OOF molecule for {task}")
            joined = c.set_index("smiles").join(
                b.set_index("smiles")[["outer_fold", "scaffold", "y_true", "y_pred"]]
                .rename(columns={"outer_fold": "base_fold", "scaffold": "base_scaffold",
                                 "y_true": "base_y", "y_pred": "base_pred"}),
                how="inner", validate="one_to_one")
            if len(joined) != len(c) or len(joined) != len(b):
                raise RuntimeError(f"OOF molecule sets differ for {task}/{base_name}")
            if not np.array_equal(joined.outer_fold.astype(int), joined.base_fold.astype(int)):
                raise RuntimeError(f"Outer fold assignments differ for {task}/{base_name}")
            if not np.allclose(joined.y_true.to_numpy(dtype=float), joined.base_y.to_numpy(dtype=float),
                               atol=1e-6, rtol=0):
                raise RuntimeError(f"Labels differ for {task}/{base_name}")
            if not np.array_equal(joined.scaffold.fillna("").astype(str).to_numpy(),
                                  joined.base_scaffold.fillna("").astype(str).to_numpy()):
                raise RuntimeError(f"Scaffold assignment differs for {task}/{base_name}")

            y = joined.y_true.to_numpy(dtype=float)
            pc = joined.y_pred.to_numpy(dtype=float)
            pb = joined.base_pred.to_numpy(dtype=float)
            delta = rmse(y, pc) - rmse(y, pb)
            scaffolds = joined.scaffold.fillna("").astype(str).to_numpy()
            groups = np.unique(scaffolds)
            idx_by_group = {g: np.flatnonzero(scaffolds == g) for g in groups}
            rng = np.random.default_rng(args.seed + 7 * ti + (0 if base_name == "matched_2d" else 1))
            boot = np.empty(args.bootstrap, dtype=float)
            for k in range(args.bootstrap):
                draw = rng.choice(groups, size=len(groups), replace=True)
                idx = np.concatenate([idx_by_group[g] for g in draw])
                boot[k] = rmse(y[idx], pc[idx]) - rmse(y[idx], pb[idx])
            lo, hi = np.quantile(boot, [0.025, 0.975])
            summaries.append({
                "task": task, "candidate_model": args.candidate_model,
                "baseline": base_name, "n": len(joined), "n_scaffolds": len(groups),
                "candidate_rmse": rmse(y, pc), "baseline_rmse": rmse(y, pb),
                "delta_rmse_candidate_minus_baseline": delta,
                "candidate_mae": float(mean_absolute_error(y, pc)),
                "baseline_mae": float(mean_absolute_error(y, pb)),
                "candidate_r2": float(r2_score(y, pc)), "baseline_r2": float(r2_score(y, pb)),
                "scaffold_bootstrap_ci95_low": float(lo),
                "scaffold_bootstrap_ci95_high": float(hi),
                "bootstrap_probability_candidate_better": float(np.mean(boot < 0)),
            })
            for fold, part in joined.groupby("outer_fold", sort=True):
                yf = part.y_true.to_numpy(dtype=float)
                fold_rows.append({
                    "task": task, "baseline": base_name, "outer_fold": int(fold),
                    "n": len(part), "candidate_rmse": rmse(yf, part.y_pred.to_numpy(dtype=float)),
                    "baseline_rmse": rmse(yf, part.base_pred.to_numpy(dtype=float)),
                    "delta_rmse": rmse(yf, part.y_pred.to_numpy(dtype=float))
                                  - rmse(yf, part.base_pred.to_numpy(dtype=float)),
                })
            if base_name == "matched_2d":
                paired_rows.append(joined.reset_index())

    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    pd.concat(paired_rows, ignore_index=True).to_csv(out / "paired_outer_oof_predictions.csv", index=False)
    audit = {
        "status": "complete", "candidate_model": args.candidate_model,
        "candidate_predictions_sha256": sha256(cpath), "matched_2d_predictions_sha256": sha256(mpath),
        "full_2d_predictions_sha256": sha256(fpath), "full_2d_model": args.full_2d_model,
        "outer_folds_per_task": 5, "paired_oof_rows_exact": True,
        "mode_selection_uses_outer_test": False,
        "outer_metric_is_paired_scaffold_holdout": True,
        "bootstrap_replicates": args.bootstrap,
        "limitation": "Internal ChEMBL scaffold cross-validation; not external or prospective validation.",
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
