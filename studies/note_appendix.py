"""Write docs/appendix.md (tables for the research note) from reports/results.json."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "reports" / "results.json").read_text())
t = pd.DataFrame(r["sim"]["by_strategy"])
names = {"market": "Market", "twap": "TWAP", "vwap": "VWAP", "pov": "POV", "ac_k0.05": "Almgren-Chriss (k=0.05)",
         "passive_twap": "Passive TWAP", "signal_aware": "Signal-aware"}
out = ["## Appendix A. Median shortfall by strategy and size (bps, base fill model, 95% interval)\n"]
for fees, label in (("retail", "Retail fees, 10/10 bps"), ("low_fee", "Low-fee schedule, 0/2 bps")):
    out.append(f"**{label}**\n")
    for sym in ("BTCUSDT", "ETHUSDT"):
        d = t[(t.fees == fees) & (t.symbol == sym) & t.strategy.isin(names)]
        sizes = sorted(d.notional.unique())
        head = "| Strategy | " + " | ".join(f"${s / 1e6:g}M" if s >= 1e6 else f"${s / 1e3:g}k" for s in sizes) + " |"
        out += [f"*{sym}*\n", head, "|---" * (len(sizes) + 1) + "|"]
        for k, v in names.items():
            row = [d[(d.strategy == k) & (d.notional == s)].iloc[0] for s in sizes]
            out.append(f"| {v} | " + " | ".join(f"{x['median']:.2f} ({x['median_lo']:.2f} to {x['median_hi']:.2f})" for x in row) + " |")
        out.append("")
out.append("## Appendix B. Where the cost goes (mean bps, low-fee schedule, base fill model, largest order size)\n")
out += ["| Symbol | Strategy | Spread | Impact | Delay | Fees | Total | Passive share |", "|---|---|---|---|---|---|---|---|"]
for sym, n in (("BTCUSDT", 5e6), ("ETHUSDT", 2e6)):
    for k in ("market", "twap", "passive_twap", "signal_aware"):
        x = t[(t.fees == "low_fee") & (t.symbol == sym) & (t.notional == n) & (t.strategy == k)].iloc[0]
        out.append(f"| {sym} | {names[k]} | {x['spread']:.2f} | {x['impact']:.2f} | {x['delay']:.2f} | {x['fee']:.2f} | "
                   f"{x['mean']:.2f} | {x['passive_frac']:.0%} |")
out.append("\nDelay is positive for the passive strategies because the unfilled part of a resting order tends to be "
           "the part the market moved away from: adverse selection shows up as timing cost.\n")
c = r["cost_curve"]
out.append("## Appendix C. Reconstructed taker orders by size (training days)\n")
for sym in ("BTCUSDT", "ETHUSDT"):
    b = pd.DataFrame(c[sym]["bins"])
    out += [f"*{sym}*\n", "| Size bucket (USD) | Orders | Mean walk (bps) | 95th percentile (bps) |", "|---|---|---|---|"]
    for _, x in b.iterrows():
        lo, hi = x["bucket"].strip("[)").split(", ")
        out.append(f"| {float(lo):,.0f} to {float(hi):,.0f} | {int(x['n']):,} | {x['mean']:.3f} | {x['p95']:.3f} |")
    out.append("")
(ROOT / "docs" / "appendix.md").write_text("\n".join(out) + "\n")
print("wrote docs/appendix.md")
