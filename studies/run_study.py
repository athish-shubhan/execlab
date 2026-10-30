"""ExecLab study: every number and chart in the README comes from this script.

    python studies/run_study.py                    # full run on ../data/binance (about 25 min)
    python studies/run_study.py --sample           # checked-in 2-hour sample (about 1 min)

Walk-forward by day: day 1 calibrates and trains for day 2; days 1-2 for day 3.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sqlite3
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from execlab import quality
from execlab.data import Trades, quote_grid, read_agg_trades, read_sample, taker_orders
from execlab.experiment import calibrate, concat, simulate_day, split_days, train_models
from execlab.features import FEATURE_GROUPS, FEATURES, build, targets
from execlab.models import _xgb, fit_baseline, per_sample_loss, tune_xgb
from skeptic.boot import block_bootstrap_by_group
from skeptic.ledger import Ledger
from skeptic.stats import deflated_sharpe_ratio, probabilistic_sharpe_ratio

ROOT = Path(__file__).resolve().parents[1]
BLUE, LIGHT, GREY, RED, GREEN = "#1f5fa8", "#9fbbe0", "#8a8a8a", "#b23a3a", "#2e7d4f"
FEES = {"retail": (10.0, 10.0), "low_fee": (0.0, 2.0)}
DT, HORIZON = 5, 900
NOTIONALS = {"BTCUSDT": [2.5e5, 1e6, 5e6], "ETHUSDT": [1e5, 5e5, 2e6]}
AC_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)


def fmt_usd(x: float) -> str:
    return f"${x / 1e6:g}M" if x >= 1e6 else f"${x / 1e3:g}k"


# ----------------------------------------------------------------------------- data
def load(sample: bool) -> dict[str, Trades]:
    if sample:
        return {p.name.split("_")[0].upper(): read_sample(p) for p in sorted((ROOT / "sample").glob("*_trades.csv.gz"))}
    out = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        fs = sorted(glob.glob(str(ROOT.parent / "data" / "binance" / f"{sym}-aggTrades-*.zip")))
        if fs:
            out[sym] = read_agg_trades(fs, sym)
    return out


# ----------------------------------------------------------------------------- fact sheet
def factsheet(trs: dict[str, Trades], out: Path) -> pd.DataFrame:
    rows = []
    for sym, tr in trs.items():
        q = quote_grid(tr, 1_000_000)
        r1m = quote_grid(tr, 60_000_000)
        hr_q = (q.t // 3_600_000_000) % 24
        hr_m = (r1m.t[1:] // 3_600_000_000) % 24
        rv = np.diff(np.log(r1m.mid)) * 1e4
        hr_t = (tr.ts // 3_600_000_000) % 24
        n_days = len(np.unique(tr.ts // 86_400_000_000))
        for h in range(24):
            mt = hr_t == h
            if mt.sum() == 0 or (hr_m == h).sum() < 2:
                continue
            rows.append({"symbol": sym, "hour_utc": h,
                         "spread_bps": float(q.spread_bps[hr_q == h].mean()),
                         "vol_1m_bps": float(rv[hr_m == h].std()),
                         "notional_musd_per_hour": float((tr.qty[mt] * tr.price[mt]).sum() / 1e6 / n_days),
                         "trades_per_second": float(mt.sum() / 3600 / n_days),
                         "buy_share": float((tr.side[mt] > 0).mean())})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for sym, c in zip(trs, (BLUE, RED), strict=False):
        d = df[df.symbol == sym]
        axes[0].plot(d.hour_utc, d.notional_musd_per_hour, color=c, label=sym)
        axes[1].plot(d.hour_utc, d.vol_1m_bps, color=c)
        axes[2].plot(d.hour_utc, d.trades_per_second, color=c)
    for ax, t in zip(axes, ("traded notional ($M per hour)", "1-minute volatility (bps)",
                            "trades per second"), strict=True):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("hour (UTC)")
        ax.set_xticks([0, 6, 12, 18, 23])
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Intraday profile, Binance spot, 27-29 Oct 2026 (the spread is usually one tick, so it is omitted)",
                 fontsize=9.5, y=1.04)
    fig.savefig(out / "factsheet.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(out / "factsheet.csv", index=False)
    return df


# ----------------------------------------------------------------------------- cost curve
def cost_curve(trs: dict[str, Trades], cals: dict, out: Path) -> dict:
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, (sym, tr) in zip(axes, trs.items(), strict=False):
        cal = cals[sym]
        b = cal.walk.bins
        b = b[b["n"] >= 5]
        x = b["notional_mid"].to_numpy()
        ax.plot(x, b["mean"], "o-", color=BLUE, ms=4, label="mean walk cost (observed taker orders)")
        ax.plot(x, b["p95"], "s--", color=LIGHT, ms=3, label="95th percentile")
        grid = np.logspace(3, np.log10(x.max() * 3), 60)
        ax.plot(grid, cal.walk(grid), color=GREY, lw=1, label="curve used by the simulator (interpolated)")
        sig_d = cal.sigma_1m_bps * np.sqrt(1440)
        ax.plot(grid, sig_d * np.sqrt(grid / cal.adv_usd), color=RED, lw=1, ls=":",
                label="square-root reference, Y=1 (metaorder)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("order notional (USD)")
        ax.set_ylabel("cost beyond the touch (bps)")
        ax.set_title(sym, fontsize=10)
        style(ax)
        over = b[b["p95"] > 1.0]
        uncomfortable = float(over["notional_mid"].iloc[0]) if len(over) else None
        res[sym] = {"powerlaw_a_over_30k": cal.walk.a, "powerlaw_b_over_30k": cal.walk.b,
                    "bins": b.assign(bucket=b["bucket"].astype(str)).to_dict(orient="records"),
                    "uncomfortable_bucket": str(over["bucket"].iloc[0]) if len(over) else None,
                    "uncomfortable_notional": uncomfortable,
                    "max_observed_order": float(taker_orders(tr)["notional"].max()),
                    "response_exponent": cal.response_exponent, "response_ci": list(cal.response_ci),
                    "lam_bps_per_musd": cal.lam, "sigma_1m_bps": cal.sigma_1m_bps,
                    "adv_musd": cal.adv_usd / 1e6, "queue_base_units": cal.queue}
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Immediate cost of a market order by size: sweeps reconstructed from aggTrades", fontsize=10)
    fig.savefig(out / "cost_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------- ML diagnostics
def ml_diagnostics(days: dict[str, dict[str, Trades]], cals_last: dict, out: Path, n_trials: int,
                   ledger: Ledger) -> dict:
    """Train on all but the last day, test on the last day. Direction, fill and cross-asset."""
    res: dict = {}
    data = {}
    for sym, dd in days.items():
        keys = sorted(dd)
        train, test = concat([dd[k] for k in keys[:-1]]), dd[keys[-1]]
        mats = []
        for tr in (train, test):
            q = quote_grid(tr, 1_000_000)
            x, y = build(tr, q), targets(tr, q, queue=cals_last[sym].queue)
            ok = (y.notna().all(axis=1) & x.notna().all(axis=1)).to_numpy()
            mats.append((x[ok], y[ok]))
        data[sym] = mats
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for sym, ((xtr, ytr), (xte, yte)) in data.items():
        r: dict = {}
        sub = slice(None, None, 2)  # every 2nd second for the direction models (speed)
        for h in (5, 1, 30):  # tune on the 5s target first, reuse its parameters for 1s and 30s
            ytr_h = (ytr[f"dir_{h}"] + 1).to_numpy()
            yte_h = (yte[f"dir_{h}"] + 1).to_numpy()
            base = fit_baseline(xtr.iloc[sub], ytr_h[sub], "multiclass", FEATURES)
            if h == 5:
                xgbm, best, _ = tune_xgb(xtr.iloc[sub], ytr_h[sub], "multiclass", FEATURES, 5,
                                         n_trials, ledger, f"diag_{sym}_dir5")
                r["dir_best_params"] = best
            else:
                xgbm = type(base)("xgb", _xgb("multiclass", r["dir_best_params"]).fit(
                    xtr[FEATURES].to_numpy()[sub], ytr_h[sub].astype(int)), "multiclass", FEATURES)
            pb, px = base.predict(xte), xgbm.predict(xte)
            prior = np.bincount(ytr_h.astype(int), minlength=3) / len(ytr_h)
            r[f"dir_{h}"] = {
                "logloss_prior": float(per_sample_loss("multiclass", yte_h, np.tile(prior, (len(yte_h), 1))).mean()),
                "logloss_logit": float(per_sample_loss("multiclass", yte_h, pb).mean()),
                "logloss_xgb": float(per_sample_loss("multiclass", yte_h, px).mean()),
                "acc_logit": float((pb.argmax(1) == yte_h).mean()),
                "acc_xgb": float((px.argmax(1) == yte_h).mean()),
                "acc_majority": float((yte_h == prior.argmax()).mean()),
            }
            if h == 1:
                # negative result: trade the 1s signal as a taker
                edge = px[:, 2] - px[:, 0]
                thr = np.quantile(np.abs(edge), 0.99)
                pos = np.where(edge > thr, 1, np.where(edge < -thr, -1, 0))
                gain = pos * yte["dm_1"].to_numpy()
                traded = pos != 0
                r["one_second_signal"] = {
                    "share_of_seconds_traded": float(traded.mean()),
                    "hit_rate": float((gain[traded] > 0).mean()),
                    "mean_gross_bps_per_trade": float(gain[traded].mean()),
                    "gross_ci": list(block_bootstrap_by_group(
                        gain[traded], (xte.index.to_numpy()[traded] // 3_600_000_000), np.mean,
                        n_boot=500).__dict__[k] for k in ("lower", "upper")),
                    "round_trip_fee_bps": {k: 2 * v[1] for k, v in FEES.items()},
                }
        # fill probability: logit vs tuned xgb, calibration
        ytr_f, yte_f = ytr["fill_buy"].to_numpy(), yte["fill_buy"].to_numpy()
        base = fit_baseline(xtr, ytr_f, "binary", FEATURES)
        xgbm, best_f, _ = tune_xgb(xtr, ytr_f, "binary", FEATURES, 5, n_trials, ledger, f"diag_{sym}_fill")
        pb, px = base.predict(xte), xgbm.predict(xte)
        r["fill_buy"] = {"base_rate_test": float(yte_f.mean()),
                         "logloss_prior": float(per_sample_loss("binary", yte_f, np.full(len(yte_f), ytr_f.mean())).mean()),
                         "logloss_logit": float(per_sample_loss("binary", yte_f, pb).mean()),
                         "logloss_xgb": float(per_sample_loss("binary", yte_f, px).mean())}
        bins = np.linspace(0, 1, 11)
        for p, c, lab in ((pb, LIGHT, "logistic"), (px, BLUE, "XGBoost")):
            k = np.clip(np.digitize(p, bins) - 1, 0, 9)
            cal = pd.DataFrame({"p": p, "y": yte_f, "k": k}).groupby("k").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
            cal = cal[cal.n >= 50]
            ax = axes[0] if sym == "BTCUSDT" else axes[1]
            ax.plot(cal.p, cal.y, "o-", ms=3, color=c, label=lab)
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":")
        ax.set_title(f"{sym}: passive bid filled within 5s (test day)", fontsize=9.5)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.legend(frameon=False, fontsize=8)
        style(ax)
        # ablation by feature group (drop one group, fixed best params)
        abl = {}
        for g, cols in FEATURE_GROUPS.items():
            keep = [f for f in FEATURES if f not in cols]
            m = _xgb("binary", best_f).fit(xtr[keep].to_numpy(), ytr_f.astype(int))
            abl[g] = float(per_sample_loss("binary", yte_f, m.predict_proba(xte[keep].to_numpy())[:, 1]).mean()
                           - r["fill_buy"]["logloss_xgb"])
        r["fill_ablation_logloss_increase"] = abl
        imp = xgbm.model.get_booster().get_score(importance_type="gain")
        r["fill_importance_gain"] = {FEATURES[int(k[1:])]: float(v) for k, v in
                                     sorted(imp.items(), key=lambda kv: -kv[1])[:8]}
        r["_fill_model"] = xgbm
        res[sym] = r
    fig.savefig(out / "fill_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # cross-asset: BTC-trained fill model on ETH test day
    if {"BTCUSDT", "ETHUSDT"} <= set(data):
        xe, ye = data["ETHUSDT"][1]
        yf = ye["fill_buy"].to_numpy()
        p = res["BTCUSDT"]["_fill_model"].predict(xe)
        res["cross_asset_fill_logloss"] = {
            "btc_model_on_eth": float(per_sample_loss("binary", yf, p).mean()),
            "eth_model_on_eth": res["ETHUSDT"]["fill_buy"]["logloss_xgb"],
            "prior_on_eth": res["ETHUSDT"]["fill_buy"]["logloss_prior"]}
    for sym in data:
        res[sym].pop("_fill_model", None)
    return res


# ----------------------------------------------------------------------------- simulations
def simulations(days: dict[str, dict[str, Trades]], n_trials: int, ledger: Ledger, use_xgb: bool,
                out: Path) -> tuple[pd.DataFrame, dict, dict]:
    frames, perf, cals = [], {}, {}
    for sym, dd in days.items():
        keys = sorted(dd)
        for i in range(1, len(keys)):
            train = concat([dd[k] for k in keys[:i]])
            t = time.perf_counter()
            cal = calibrate(train, DT)
            cache = out / "models" / f"{sym}_{keys[i]}.pkl"
            if cache.exists():  # trained models are cached so reruns skip the Optuna search
                models = pickle.loads(cache.read_bytes())
            else:
                models, _, _ = train_models(train, cal, DT, n_trials, ledger, f"policy_{sym}_{keys[i]}", use_xgb)
                cache.parent.mkdir(exist_ok=True)
                cache.write_bytes(pickle.dumps(models))
            print(f"  {sym} {keys[i]}: calibrated and trained in {time.perf_counter() - t:.0f}s", flush=True)
            for lab, kw in (("main", {}), ("no_self_impact", {"lam_mult": 0.0}),
                            ("permanent_impact", {"half_life_s": float("inf")})):
                fills = ("pessimistic", "base", "optimistic") if lab == "main" else ("base",)
                df, pf = simulate_day(dd[keys[i]], cal, models, DT, HORIZON, NOTIONALS[sym], 1200, FEES,
                                      AC_SWEEP if lab == "main" else [0.0], fill_models=fills, **kw)
                frames.append(df.assign(symbol=sym, impact_case=lab))
                if lab == "main":
                    perf[f"{sym}_{keys[i]}"] = pf
            cals[sym] = cal
    return pd.concat(frames, ignore_index=True), perf, cals


def summarise(sim: pd.DataFrame, n_search_trials: int) -> dict:
    res: dict = {}
    main = sim[sim.impact_case == "main"].copy()
    main["hour"] = main["t0"] // 3_600_000_000
    # shortfall by strategy and size (base fill, both fee schedules)
    tab = []
    for (sym, fees, notional, strat), g in main[main.fill_model == "base"].groupby(
            ["symbol", "fees", "notional", "strategy"]):
        v = g["total_bps"].to_numpy()
        bm = block_bootstrap_by_group(v, g["hour"].to_numpy(), np.median, n_boot=400)
        tab.append({"symbol": sym, "fees": fees, "notional": notional, "strategy": strat, "n": len(v),
                    "median": bm.estimate, "median_lo": bm.lower, "median_hi": bm.upper,
                    "p95": float(np.percentile(v, 95)), "mean": float(v.mean()), "std": float(v.std()),
                    "spread": g["spread_bps"].mean(), "impact": g["impact_bps"].mean(),
                    "delay": g["delay_bps"].mean(), "fee": g["fee_bps"].mean(),
                    "opportunity": g["opportunity_bps"].mean(), "passive_frac": g["passive_frac"].mean()})
    res["by_strategy"] = tab
    # paired improvements: signal_aware and passive_twap vs twap, per fill model and fee schedule
    keys = ["symbol", "day", "t0", "side", "notional", "fees", "fill_model"]
    piv = main.pivot_table(index=keys, columns="strategy", values="total_bps").reset_index()
    piv["hour"] = piv["t0"] // 3_600_000_000
    paired = []
    for (sym, fees, fm), g in piv.groupby(["symbol", "fees", "fill_model"]):
        for base, comp in (("twap", "signal_aware"), ("twap", "passive_twap"), ("passive_twap", "signal_aware")):
            if comp not in g or base not in g:
                continue
            d = (g[base] - g[comp]).to_numpy()  # positive = comp is cheaper than base
            b_mean = block_bootstrap_by_group(d, g["hour"].to_numpy(), np.mean, n_boot=1000)
            b_med = block_bootstrap_by_group(d, g["hour"].to_numpy(), np.median, n_boot=1000)
            paired.append({"symbol": sym, "fees": fees, "fill_model": fm, "vs": comp, "base": base, "n": len(d),
                           "mean_bps": b_mean.estimate, "mean_lo": b_mean.lower, "mean_hi": b_mean.upper,
                           "median_bps": b_med.estimate, "median_lo": b_med.lower, "median_hi": b_med.upper,
                           "share_better": float((d > 0).mean()),
                           "psr": probabilistic_sharpe_ratio(d) if d.std() > 0 else float("nan"),
                           "dsr": deflated_sharpe_ratio(d, max(n_search_trials, 1)) if d.std() > 0 else float("nan")})
    res["paired"] = paired
    # sensitivity to the self-impact assumption: market vs twap
    sens = []
    for (case, sym, fees, notional, strat), g in sim[(sim.fill_model == "base") & sim.strategy.isin(["market", "twap"])].groupby(
            ["impact_case", "symbol", "fees", "notional", "strategy"]):
        sens.append({"impact_case": case, "symbol": sym, "fees": fees, "notional": notional,
                     "strategy": strat, "mean": g["total_bps"].mean(), "impact": g["impact_bps"].mean()})
    res["impact_sensitivity"] = sens
    # AC frontier
    ac = main[(main.fill_model == "base") & main.strategy.str.startswith("ac") & (main.fees == "retail")]
    res["ac_frontier"] = (ac.groupby(["symbol", "notional", "strategy"])["total_bps"]
                          .agg(["mean", "std"]).reset_index().to_dict(orient="records"))
    return res


def sim_charts(res: dict, out: Path) -> None:
    tab = pd.DataFrame(res["by_strategy"])
    order = ["market", "twap", "vwap", "pov", "passive_twap", "signal_aware"]
    names = {"market": "market", "twap": "TWAP", "vwap": "VWAP", "pov": "POV", "passive_twap": "passive TWAP",
             "signal_aware": "signal-aware"}
    syms = sorted(tab.symbol.unique())
    fig, axes = plt.subplots(1, len(syms), figsize=(5.4 * len(syms), 3.6), squeeze=False)
    for ax, sym in zip(axes[0], syms, strict=True):
        d = tab[(tab.symbol == sym) & (tab.fees == "low_fee")]
        sizes = sorted(d.notional.unique())
        w = 0.8 / len(sizes)
        for i, n in enumerate(sizes):
            dd = d[d.notional == n].set_index("strategy").reindex(order)
            x = np.arange(len(order)) + (i - (len(sizes) - 1) / 2) * w
            ax.bar(x, dd["median"], width=w, color=[LIGHT, BLUE, "#163f73"][i], label=fmt_usd(n))
            ax.errorbar(x, dd["median"], yerr=[dd["median"] - dd["median_lo"], dd["median_hi"] - dd["median"]],
                        fmt="none", ecolor="#333", lw=0.8, capsize=1.5)
        ax.set_xticks(np.arange(len(order)), [names[o] for o in order], fontsize=8, rotation=15)
        ax.set_ylabel("median shortfall vs arrival mid (bps)")
        ax.set_title(f"{sym}: 15-minute parent orders, low-fee schedule", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, title="parent order size", title_fontsize=8, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), ncol=3)
        style(ax)
    fig.savefig(out / "shortfall_by_strategy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    p = pd.DataFrame(res["paired"])
    p = p[(p.vs == "signal_aware") & (p.base == "twap")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    fmods = ["pessimistic", "base", "optimistic"]
    for ax, fees in zip(axes, ("retail", "low_fee"), strict=True):
        for j, sym in enumerate(syms):
            d = p[(p.symbol == sym) & (p.fees == fees)].set_index("fill_model").reindex(fmods)
            x = np.arange(3) + (j - 0.5) * 0.3
            ax.errorbar(x, d.mean_bps, yerr=[d.mean_bps - d.mean_lo, d.mean_hi - d.mean_bps], fmt="o",
                        color=[BLUE, RED][j], capsize=3, label=sym)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xticks(range(3), fmods)
        ax.set_xlabel("fill model")
        ax.set_title(f"{'Retail fees (10/10 bps)' if fees == 'retail' else 'Low-fee schedule (0/2 bps)'}",
                     fontsize=9.5)
        style(ax)
    axes[0].set_ylabel("TWAP minus signal-aware (bps)\npositive = signal-aware cheaper")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Paired improvement of the signal-aware policy over TWAP, mean with 95% block-bootstrap interval",
                 fontsize=9.5)
    fig.savefig(out / "paired_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ac = pd.DataFrame(res["ac_frontier"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (sym, n), g in ac.groupby(["symbol", "notional"]):
        if n != max(NOTIONALS[sym]):
            continue
        g = g.assign(k=g.strategy.str.extract(r"k([\d.e-]+)")[0].astype(float).fillna(0)).sort_values("k")
        ax.plot(g["std"], g["mean"], "o-", ms=4, label=f"{sym} {fmt_usd(n)}")
        ax.annotate("TWAP", (g["std"].iloc[0], g["mean"].iloc[0]), fontsize=7, xytext=(4, 2),
                    textcoords="offset points")
    ax.set_xlabel("std of shortfall across orders (bps)")
    ax.set_ylabel("mean shortfall (bps)")
    ax.set_title("Almgren-Chriss risk-aversion sweep (retail fees)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.savefig(out / "ac_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--reuse-ml", action="store_true", help="keep the ML diagnostics from the last results.json")
    ap.add_argument("--reuse-sims", action="store_true",
                    help="reuse reports/simulations.csv.gz and the ledger; recalibrate only (minutes, not half an hour)")
    a = ap.parse_args()
    out = ROOT / "reports" / ("sample" if a.sample else "")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trs = load(a.sample)
    days = {s: split_days(t) for s, t in trs.items()}
    if a.sample:  # the sample is two hours of one day: treat each hour as a "day"
        days = {s: {f"h{i}": t.slice_time(int(t.ts[0]) + i * 3_600_000_000, int(t.ts[0]) + (i + 1) * 3_600_000_000)
                    for i in range(2)} for s, t in trs.items()}
    ledger = Ledger(out / "ledger.sqlite")
    results: dict = {"data": {}}
    for sym, t in trs.items():
        results["data"][sym] = quality.trades_report(t, quote_grid(t, 1_000_000))
    (out / "data_quality.md").write_text("# Data quality report\n\n" + quality.to_markdown(list(results["data"].values())) + "\n")
    print("quality done", flush=True)
    results["factsheet"] = factsheet(trs, out).to_dict(orient="records")
    if a.reuse_sims and (out / "simulations.csv.gz").exists():
        sim = pd.read_csv(out / "simulations.csv.gz")
        prev = out / "performance.json"
        perf = json.loads(prev.read_text()) if prev.exists() else {}
        cals = {s: calibrate(concat([dd[k] for k in sorted(dd)[:-1]]), DT) for s, dd in days.items()}
    else:
        sim, perf, cals = simulations(days, a.trials, ledger, use_xgb=not a.sample, out=out)
        sim.to_csv(out / "simulations.csv.gz", index=False)
        (out / "performance.json").write_text(json.dumps(perf, indent=2))
    results["performance"] = perf
    results["cost_curve"] = cost_curve(trs, cals, out)
    print("simulations done", flush=True)
    with sqlite3.connect(out / "ledger.sqlite") as c:
        n_search = int(c.execute("SELECT COUNT(*) FROM trials WHERE study LIKE 'policy_%'").fetchone()[0])
    results["n_search_trials"] = n_search
    results["sim"] = summarise(sim, n_search)
    sim_charts(results["sim"], out)
    prev_res = out / "results.json"
    if a.reuse_ml and prev_res.exists() and "ml" in json.loads(prev_res.read_text()):
        results["ml"] = json.loads(prev_res.read_text())["ml"]
    elif not a.sample:
        results["ml"] = ml_diagnostics(days, cals, out, a.trials, ledger)
    bench = out / "benchmark.json"
    if bench.exists():
        results["benchmark"] = json.loads(bench.read_text())
    results["runtime_seconds"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(results, indent=2, default=lambda o: float(o) if np.isscalar(o) else str(o)))
    print(f"done in {results['runtime_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
