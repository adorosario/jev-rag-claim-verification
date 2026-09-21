#!/usr/bin/env python3
"""Paired Jev vs gpt-6-astra analysis on identical dev examples.

Every number in medium/article.md comes from the JSON this writes. Nothing is
typed by hand (CLAUDE.md rule 4). Statistics: stratified bootstrap CIs and
McNemar on the paired errors (spec §19).

    docker compose run --rm dev uv run python scripts/verify/paired_analysis.py
"""
import json, sys
import yaml
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data.load_aggrefact import load_dev_frame  # noqa: E402

# All three runs are §28/§29 compliant and use the P2b prompts (both models name
# entity errors). Superseded runs are kept on disk but never read here:
#   paired-astra-vs-jev-20260921      v1, discarded: our 16-token cap caused 84 failures
#   paired-astra-vs-jev-20260921-v2   pre-P2b prompt
#   paired-astra-highthink-20260921   pre-P2b prompt
#   runs/explore-jev-dev              exploratory Jev, no manifest, reduced schema
JEV = REPO / "runs/paired-jev-20260921/predictions_jev.jsonl"
ASTRA = REPO / "runs/paired-astra-lowthink-20260921-p2b/predictions_gpt6_astra.jsonl"
ASTRA_HIGH = REPO / "runs/paired-astra-highthink-20260921-p2b/predictions_gpt6_astra.jsonl"
OUT = REPO / "medium/generated/numbers.json"
SEED = 20260918

def _bacc_fast(gold, pred):
    """Balanced accuracy without sklearn call overhead; None if a class is absent."""
    pos = gold == 1
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return None
    return 0.5 * (float((pred[pos] == 1).sum()) / npos + float((pred[~pos] == 0).sum()) / nneg)


def bacc_by_ds(df, col):
    """Same definition as the bootstrap uses. A single-class dataset is an error,
    not a silently reweighted number from sklearn."""
    out = {}
    for d, t in df.groupby("ds"):
        b = _bacc_fast(t.gold.to_numpy(), t[col].to_numpy())
        if b is None:
            raise ValueError(f"dataset {d!r} has a single gold class; balanced accuracy is undefined")
        out[d] = b
    return out


def mean_bacc(df, col):
    return float(np.mean(list(bacc_by_ds(df, col).values())))


def _boot(df, n, which):
    """One stratified PAIRED bootstrap: the same resampled rows score both models,
    so the delta CI keeps the pairing (spec §19)."""
    rng = np.random.default_rng(SEED)
    groups = [(t.gold.to_numpy(), t.jev.to_numpy(), t.astra.to_numpy()) for _, t in df.groupby("ds")]
    out = np.empty(n)
    for i in range(n):
        vj, va = [], []
        for g, j, a in groups:
            idx = rng.integers(0, len(g), len(g))
            gg = g[idx]
            bj = _bacc_fast(gg, j[idx])
            if bj is None:      # a single-class resample scores neither model
                continue
            vj.append(bj); va.append(_bacc_fast(gg, a[idx]))
        mj, ma = float(np.mean(vj)), float(np.mean(va))
        out[i] = mj if which == "jev" else ma if which == "astra" else mj - ma
    return out


def boot_ci(df, col, n=10000):
    b = _boot(df, n, col)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_delta_ci(df, a, b, n=10000):
    d = _boot(df, n, "delta")
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def main() -> int:
    dev = load_dev_frame()
    # errata E18: a few (doc, claim) pairs appear twice in dev, some with conflicting
    # labels. Those are unscoreable, so they are excluded rather than last-wins.
    _g = dev.labels.groupby("example_id")["label"].agg(["nunique", "first"])
    ambiguous = set(_g.index[_g["nunique"] > 1])
    gold = {e: int(v) for e, (nu, v) in _g[["nunique", "first"]].iterrows() if nu == 1}
    jev = {}
    for l in JEV.read_text().splitlines():             # append-only: last row wins
        if l.strip(): r = json.loads(l); jev[r["example_id"]] = r
    astra = {}
    for l in ASTRA.read_text().splitlines():           # append-only: last row wins
        if l.strip(): r = json.loads(l); astra[r["example_id"]] = r

    rows = []
    for eid, j in jev.items():
        a = astra.get(eid)
        if a is None or j["status"] != "ok" or a["status"] != "ok": continue
        if eid in ambiguous: continue
        if eid not in gold: raise KeyError(f"{eid} is absent from dev gold: wrong split or a stale run?")
        rows.append(dict(eid=eid, ds=j["dataset"], gold=gold[eid],
                         jev=1 if j["label"] == "SUPPORTED" else 0,
                         astra=1 if a["label"] == "SUPPORTED" else 0,
                         jev_p=j["p_supported"], jev_ms=j["latency_ms"], astra_ms=a["latency_ms"],
                         jev_in=j["input_tokens"] or 0, astra_in=a["input_tokens"] or 0,
                         astra_out=a["output_tokens"] or 0))
    df = pd.DataFrame(rows).reset_index(drop=True)  # positional == label, for the cascade
    n_astra_rows = len(astra); n_astra_ok = sum(1 for r in astra.values() if r["status"] == "ok")
    n_jev_rows = len(jev); n_jev_ok = sum(1 for r in jev.values() if r["status"] == "ok")

    # Operational view (CLAUDE.md rule 6): any example either model failed on is
    # scored as WRONG for the model that failed, never silently dropped.
    op_rows = []
    for eid, j in jev.items():
        a = astra.get(eid)
        if a is None or eid in ambiguous or eid not in gold: continue
        g = gold[eid]
        op_rows.append(dict(ds=j["dataset"], gold=g,
                            jev=(1 if j["label"] == "SUPPORTED" else 0) if j["status"] == "ok" else 1 - g,
                            astra=(1 if a["label"] == "SUPPORTED" else 0) if a["status"] == "ok" else 1 - g))
    op = pd.DataFrame(op_rows)
    operational = dict(n=int(len(op)),
                       jev_mean_bacc=mean_bacc(op, "jev"), astra_mean_bacc=mean_bacc(op, "astra"))

    per_j, per_a = bacc_by_ds(df, "jev"), bacc_by_ds(df, "astra")
    mj, ma = mean_bacc(df, "jev"), mean_bacc(df, "astra")
    cj, ca = boot_ci(df, "jev"), boot_ci(df, "astra")
    dmean, dlo, dhi = boot_delta_ci(df, "jev", "astra")

    jw = int(((df.jev == df.gold) & (df.astra != df.gold)).sum())
    aw = int(((df.astra == df.gold) & (df.jev != df.gold)).sum())
    mc = mcnemar([[int(((df.jev == df.gold) & (df.astra == df.gold)).sum()), jw],
                  [aw, int(((df.jev != df.gold) & (df.astra != df.gold)).sum())]], exact=True)
    agree = float((df.jev == df.astra).mean())

    # Prices come from the hashed pricing snapshot, never from this file (CLAUDE.md rule 4).
    prices = yaml.safe_load((REPO / "configs/pricing_2026-09-20.yaml").read_text())["prices_per_mtok_usd"]
    jev_in_price = float(prices["jev-1.13.0"]["input"])
    astra_in = float(prices["gpt-6-astra"]["input"])
    astra_out = float(prices["gpt-6-astra"]["output"])
    assert float(prices["jev-1.13.0"]["output"]) == 0.0, "Jev output pricing changed; update the cost model"
    jev_cost_1k = float(df.jev_in.mean() * 1000 / 1e6 * jev_in_price)
    astra_cost_1k = float(df.astra_in.mean() * 1000 / 1e6 * astra_in + df.astra_out.mean() * 1000 / 1e6 * astra_out)

    def fvr(col):    # gold=0 predicted 1
        neg = df[df.gold == 0]; return float((neg[col] == 1).mean())
    def vprec(col):  # precision of "supported"
        pos = df[df[col] == 1]; return float((pos.gold == 1).mean())

    # confidence gating on Jev's native probability
    gate = []
    for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99):
        acc = df[(df.jev_p >= t) | (df.jev_p <= 1 - t)]
        if len(acc) == 0: continue
        per = [b for _, x in acc.groupby("ds")
               if (b := _bacc_fast(x.gold.to_numpy(), x["jev"].to_numpy())) is not None]
        gate.append(dict(threshold=t, coverage=float(len(acc) / len(df)),
                         # macro over datasets: the same convention as the headline
                         bacc=float(np.mean(per)) if per else None,
                         bacc_pooled=float(balanced_accuracy_score(acc.gold, acc.jev)) if acc.gold.nunique() > 1 else None,
                         n_datasets_scored=len(per),
                         verified_precision=float((acc[acc.jev == 1].gold == 1).mean()) if (acc.jev == 1).any() else None,
                         n=int(len(acc))))
    # cascade: Jev when confident, Astra otherwise
    casc = []
    for g in gate:
        t = g["threshold"]; conf = (df.jev_p >= t) | (df.jev_p <= 1 - t)
        final = pd.Series(np.where(conf, df.jev, df.astra), index=df.index)
        esc = float((~conf).mean())
        e_rows = df[~conf]
        astra_esc_1k = float(e_rows.astra_in.mean() * 1000 / 1e6 * astra_in
                             + e_rows.astra_out.mean() * 1000 / 1e6 * astra_out) if len(e_rows) else 0.0
        casc.append(dict(threshold=t, escalated=esc,
                         bacc=float(np.mean([_bacc_fast(t2.gold.to_numpy(), final.loc[t2.index].to_numpy())
                                             for _, t2 in df.groupby("ds") if t2.gold.nunique() > 1])),
                         # Jev runs on every claim; escalated ones additionally pay Astra,
                         # priced on the escalated subset's own (longer) token profile.
                         cost_per_1k=float(jev_cost_1k + esc * astra_esc_1k)))

    # Fairness ablation (errata E21): the same 495 items at reasoning_effort=high,
    # so nobody can say the frontier model was handicapped by a cheap setting.
    astra_high = {}
    if ASTRA_HIGH.is_file():
        for l in ASTRA_HIGH.read_text().splitlines():
            if l.strip():
                r = json.loads(l); astra_high[r["example_id"]] = r
    high = None
    if astra_high:
        hh = df.copy()
        missing = [e for e in hh.eid if e not in astra_high]
        if missing:
            raise SystemExit(f"REFUSING to write numbers.json: the high-effort ablation covers "
                             f"{len(hh) - len(missing)}/{len(hh)} examples; errata E21 needs the same items.")
        bad = [e for e in hh.eid if astra_high[e]["status"] != "ok"]
        if bad:
            raise SystemExit(f"REFUSING to write numbers.json: {len(bad)} high-effort rows are not ok.")
        if True:
            hh["high"] = [1 if astra_high[e]["label"] == "SUPPORTED" else 0 for e in hh.eid]
            per_h = bacc_by_ds(hh, "high")
            htok_in = float(np.mean([astra_high[e]["input_tokens"] or 0 for e in hh.eid]))
            htok_out = float(np.mean([astra_high[e]["output_tokens"] or 0 for e in hh.eid]))
            hms = [astra_high[e]["latency_ms"] for e in hh.eid]
            high = dict(mean_bacc=float(np.mean(list(per_h.values()))), per_dataset=per_h,
                        cost_per_1k=float(htok_in * 1000 / 1e6 * astra_in + htok_out * 1000 / 1e6 * astra_out),
                        p50_ms=float(np.median(hms)), mean_output_tokens=htok_out,
                        fvr=float((hh[hh.gold == 0].high == 1).mean()),
                        verified_precision=float((hh[hh.high == 1].gold == 1).mean()),
                        valid_rate=float(sum(1 for r in astra_high.values() if r["status"] == "ok") / len(astra_high)),
                        reasoning_effort="high", max_completion_tokens=4096)

    out = dict(
        astra_high_effort=high,
        generated_from=dict(jev=str(JEV.relative_to(REPO)), astra=str(ASTRA.relative_to(REPO)),
                            dataset="lytang/LLM-AggreFact dev", parquet_sha256=dev.parquet_sha256),
        n_paired=int(len(df)), n_datasets=int(df.ds.nunique()),
        astra_rows=n_astra_rows, astra_ok=n_astra_ok,
        jev_valid_rate=float(sum(1 for r in jev.values() if r["status"] == "ok") / len(jev)),
        astra_valid_rate=float(n_astra_ok / n_astra_rows) if n_astra_rows else None,
        jev=dict(mean_bacc=mj, ci=[cj[0], cj[1]], per_dataset=per_j, fvr=fvr("jev"),
                 verified_precision=vprec("jev"), cost_per_1k=jev_cost_1k,
                 p50_ms=float(df.jev_ms.median()), p95_ms=float(df.jev_ms.quantile(.95)),
                 mean_input_tokens=float(df.jev_in.mean())),
        astra=dict(mean_bacc=ma, ci=[ca[0], ca[1]], per_dataset=per_a, fvr=fvr("astra"),
                   verified_precision=vprec("astra"), cost_per_1k=astra_cost_1k,
                   p50_ms=float(df.astra_ms.median()), p95_ms=float(df.astra_ms.quantile(.95)),
                   mean_input_tokens=float(df.astra_in.mean()), mean_output_tokens=float(df.astra_out.mean())),
        delta=dict(mean=dmean, ci=[dlo, dhi], jev_minus_astra_pp=float((mj - ma) * 100)),
        paired=dict(agreement=agree, jev_only_correct=jw, astra_only_correct=aw,
                    mcnemar_p=float(mc.pvalue), mcnemar_statistic=float(mc.statistic),
                    mcnemar_tests="RAW per-item accuracy pooled over datasets, NOT the "
                                  "macro-averaged balanced accuracy reported in delta.",
                    raw_accuracy_favors=("jev" if jw > aw else "astra" if aw > jw else "tie")),
        # No latency ratio is emitted: the runs used different concurrency, and the
        # controlled §21 protocol has not been run (CLAUDE.md rule 10).
        ratios=dict(cost=float(astra_cost_1k / jev_cost_1k)),
        raw_accuracy=dict(jev=float((df.jev == df.gold).mean()),
                          astra=float((df.astra == df.gold).mean())),
        operational=operational,
        jev_rows=n_jev_rows, jev_ok=n_jev_ok,
        concurrency_note="Both runs used concurrency 12; per-request latency therefore "
                         "includes client-side queueing and is NOT a controlled measurement.",
        gating=gate, cascade=casc,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("n_paired", "jev", "astra", "delta", "paired", "ratios")},
                     indent=2, sort_keys=True)[:2600])
    print("\nwrote", OUT.relative_to(REPO))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
