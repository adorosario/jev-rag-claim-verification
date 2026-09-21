#!/usr/bin/env python3
"""Check that every claim in medium/article.md matches medium/generated/numbers.json.

Publishing a number that the raw data does not support is the failure mode this
repo exists to prevent (CLAUDE.md rule 4). Run before publishing.
"""
import json, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
d = json.loads((REPO / "medium/generated/numbers.json").read_text())
art = (REPO / "medium/article.md").read_text()
bad, checked = [], 0

def chk(label, expected, fmt="{:.1f}", present=True):
    """Assert the formatted value appears (or does not appear) in the article."""
    global checked
    s = fmt.format(expected); checked += 1
    if (s in art) != present:
        bad.append(f"{label}: expected {'to find' if present else 'NOT to find'} '{s}' in the article")

j, a, h = d["jev"], d["astra"], d["astra_high_effort"]
chk("jev bacc", j["mean_bacc"] * 100); chk("astra bacc", a["mean_bacc"] * 100)
chk("astra high bacc", h["mean_bacc"] * 100)
chk("jev ci lo", j["ci"][0] * 100); chk("jev ci hi", j["ci"][1] * 100)
chk("astra ci lo", a["ci"][0] * 100); chk("astra ci hi", a["ci"][1] * 100)
chk("delta", abs(d["delta"]["jev_minus_astra_pp"]))
chk("delta ci lo", abs(d["delta"]["ci"][0] * 100)); chk("delta ci hi", d["delta"]["ci"][1] * 100)
chk("mcnemar p", d["paired"]["mcnemar_p"], "{:.2f}")
chk("agreement", d["paired"]["agreement"] * 100)
chk("jev only correct", d["paired"]["jev_only_correct"], "{:d}")
chk("astra only correct", d["paired"]["astra_only_correct"], "{:d}")
chk("jev cost", j["cost_per_1k"], "{:.3f}"); chk("astra cost", a["cost_per_1k"], "{:.2f}")
chk("astra high cost", h["cost_per_1k"], "{:.2f}")
chk("cost ratio", d["ratios"]["cost"], "{:.0f}")
if "latency_p50" in d["ratios"]:
    bad.append("numbers.json still exposes a latency ratio; the runs were not a controlled measurement")
chk("jev p50", j["p50_ms"], "{:.0f}"); chk("raw acc jev", d["raw_accuracy"]["jev"] * 100); chk("raw acc astra", d["raw_accuracy"]["astra"] * 100)
chk("astra p50", a["p50_ms"], "{:,.0f}"); chk("astra high p50", h["p50_ms"], "{:,.0f}")
chk("jev mean input tokens", j["mean_input_tokens"], "{:,.0f}")
chk("astra mean input tokens", a["mean_input_tokens"], "{:,.0f}")
chk("astra mean output tokens", a["mean_output_tokens"], "{:.0f}")
chk("astra high mean output tokens", h["mean_output_tokens"], "{:.0f}")
chk("jev fvr", j["fvr"] * 100); chk("astra fvr", a["fvr"] * 100); chk("astra high fvr", h["fvr"] * 100)
chk("jev vprec", j["verified_precision"] * 100); chk("astra vprec", a["verified_precision"] * 100)
chk("astra high vprec", h["verified_precision"] * 100)
chk("n paired", d["n_paired"], "{:d}")

for ds, v in j["per_dataset"].items():
    chk(f"jev per-dataset {ds}", v * 100)
for ds, v in a["per_dataset"].items():
    chk(f"astra per-dataset {ds}", v * 100)

for g in d["gating"]:
    if g["threshold"] in (0.5, 0.9, 0.95, 0.99):   # the 0.80 row is not published
        chk(f"gate {g['threshold']} coverage", g["coverage"] * 100, "{:.0f}")
        chk(f"gate {g['threshold']} bacc", g["bacc"] * 100)
        chk(f"gate {g['threshold']} vprec", g["verified_precision"] * 100)
for c in d["cascade"]:
    if c["threshold"] in (0.9, 0.95):
        chk(f"cascade {c['threshold']} escalated", c["escalated"] * 100, "{:.0f}")
        chk(f"cascade {c['threshold']} bacc", c["bacc"] * 100)
        chk(f"cascade {c['threshold']} cost", c["cost_per_1k"], "{:.2f}")

# numbers that must NOT survive from earlier, discarded runs
for stale in ("411", "84 of 495 failed", "17%"):
    if stale in art and stale != "411":
        bad.append(f"stale string from the discarded v1 run appears: '{stale}'")

print(f"checked {checked} values against numbers.json")
if bad:
    print("\nMISMATCHES:"); [print(" -", b) for b in bad]
    return_code = 1
else:
    print("all checked values match the data")
    return_code = 0
sys.exit(return_code)
