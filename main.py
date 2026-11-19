"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01), Python 3.13 stdlib only.

Reads one case (/input/<id>.in): Exophase's earn rate for each of GTA VI's 55 PS5
trophies as the house recorded it, and forecasts each trophy's rate at the case's
target as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit
scale. Writes /output/<id>.out. Deterministic: pure arithmetic, no randomness,
no network, no third-party packages.

Model
-----
Work in logit space, where a move from 1% to 2% costs about what 33% to 50%
does -- the scale the scorer uses. Two regimes, detected from the case itself:

1. Moving history (the final case). Consecutive identical reads are collapsed --
   Exophase refreshes a game's page every few hours, so the raw 10-minute reads
   repeat the same page and would fake a huge sample size. On the deduplicated
   reads of the last 24 hours, fit each trophy a least-squares line of logit
   rate against ln(1 + hours since release). Log time damps the extrapolation:
   the final's target is only ~1 log-hour past the last read. Slopes are shrunk
   toward their tier's median slope by empirical-Bayes precision weighting, so a
   trophy whose own series is noisy borrows strength from its tier. The median
   anchors on the trophy's last observed rate and moves along the shrunk slope;
   the spread combines a random-walk term (0.5 logit per root-day, as the house
   baseline), the shrunk fit residual, and an extrapolation penalty that grows
   with how far past the data the target sits. The spread is mildly asymmetric
   (wider below the median): over a launch week, new players keep arriving and
   dilute every rate, so the downside risk dominates.

2. Flat history (the practice cases). Every read identical means the page is
   stale and a refresh is due. The rules' own launch-night record shows what a
   refresh does: players tracked jumped 638 -> 3,868 while trophies earned went
   943 -> 4,276, and nearly all of the new earns were the one trophy already at
   100% -- so every other trophy's rate fell by roughly the player-growth
   factor. The forecast therefore shifts each trophy's median down by
   1.6 * (1 - sqrt(rate)) logit units (nothing for a trophy at 100%, about
   -1.5 for a typical ~1% trophy), with a wide asymmetric spread whose long
   upside tail covers the chance that no refresh lands inside the horizon.

Quantiles are rounded to 4 decimals (half up) and forced non-decreasing.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day began 2026-11-19T00:00:00Z; the trend runs on ln(1 + hours since).
RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

# Random-walk spread of a rate's move, logit units per square-root day.
RW_SIGMA = 0.5
# Prior spread of per-trophy slopes around their tier median (empirical Bayes).
SLOPE_TAU = 0.35
# Stale-page dilution prior: logit shift of a refresh, before the rate factor.
DILUTION_SHIFT = 1.6
# Stale-page spread: (below median, above median) in logit units.
STALE_SPREAD_DOWN = 0.5
STALE_SPREAD_UP = 0.8
# Spread floor / cap for the moving-history regime.
SPREAD_MIN = 0.35
SPREAD_MAX = 2.0
# Asymmetry of the moving-history spread: downside wider (dilution risk).
SPREAD_DOWN_FACTOR = 1.15
SPREAD_UP_FACTOR = 0.9
# Safety valve on any single median move, logit units. A trophy's earn rate
# follows an S-curve as the player base progresses, so a line fitted to the
# steep part over-extrapolates: 2.5 logit still allows a 1% -> 11% or a
# 45% -> 92% move over the horizon, but blocks the most extreme blowups.
MAX_MOVE = 2.5
# Trend window: the last 24 hours of reads (falls back to all reads if too few).
WINDOW_HOURS = 24.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def log_hours(epoch):
    """ln(1 + hours since release), guarded for reads just before midnight."""
    hours = (epoch - RELEASE_EPOCH) / 3600.0
    return math.log(max(1.0 + hours, 0.01))


def dedupe_reads(history):
    """Collapse consecutive identical page states into one observation.

    Exophase only refreshes a game's counts every few hours, so most raw reads
    repeat the previous page exactly. Keeping them would fake a large sample.
    """
    out = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if out and out[-1][0] == key:
            continue
        out.append((key, read))
    return [read for _, read in out]


def ols(xs, ys):
    """Least-squares line; returns (intercept, slope, resid_std, n, sxx)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 0.0:
        return my, 0.0, 0.0, n, sxx
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    resid = math.sqrt(sse / max(n - 2, 1)) if n > 2 else 0.0
    return intercept, slope, resid, n, sxx


def median(values):
    vals = sorted(values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    checkpoint = parse_time(case["at"])
    target_x = log_hours(parse_time(case["target"]))

    if not history:
        # No data at all: a wide, neutral forecast beats a crash.
        return {"trophies": [
            {"id": t["id"], "q": [round4(inv_logit(logit(1.0) + z * 1.0)) for z in Z]}
            for t in trophies
        ]}

    reads = dedupe_reads(history)
    last_read = history[-1]
    last_rates = last_read["rates"]
    stale = len(reads) == 1

    # Fallback level for a trophy never observed: its tier's median last rate.
    tier_last = {}
    for t, r in zip(trophies, last_rates):
        if r is not None:
            tier_last.setdefault(t["tier"], []).append(r)
    global_last = [r for r in last_rates if r is not None]
    global_level = median(global_last) if global_last else 1.0

    def fallback_level(tier):
        vals = tier_last.get(tier)
        return median(vals) if vals else global_level

    rows = []
    if stale:
        # Flat history: the page is stale, a refresh is due, and a refresh
        # dilutes every non-universal trophy by roughly the player-growth
        # factor (see the module docstring for the launch-night evidence).
        for i, trophy in enumerate(trophies):
            rate = last_rates[i]
            if rate is None:
                rate = fallback_level(trophy["tier"])
            frac = clamp_pct(rate) / 100.0
            med = logit(rate) - DILUTION_SHIFT * (1.0 - math.sqrt(frac))
            qs = []
            prev = None
            for z in Z:
                spread = STALE_SPREAD_DOWN if z < 0 else STALE_SPREAD_UP
                q = round4(min(100.0, max(0.0, inv_logit(med + z * spread))))
                if prev is not None and q < prev:
                    q = prev
                qs.append(q)
                prev = q
            rows.append({"id": trophy["id"], "q": qs})
        return {"trophies": rows}

    # Moving history: per-trophy trend on log time, last 24h of distinct reads.
    window_start = checkpoint - WINDOW_HOURS * 3600.0
    window = [r for r in reads if parse_time(r["at"]) >= window_start]
    if len(window) < 3:
        window = reads
    window_x = [log_hours(parse_time(r["at"])) for r in window]
    last_x = window_x[-1]

    fits = []
    for i, trophy in enumerate(trophies):
        xs, ys = [], []
        for r, x in zip(window, window_x):
            rate = r["rates"][i]
            if rate is not None:
                xs.append(x)
                ys.append(logit(rate))
        if not xs:
            fits.append(None)
            continue
        fits.append(ols(xs, ys))

    # Tier-median slopes for shrinkage, and a global residual scale.
    tier_slopes = {}
    for trophy, fit in zip(trophies, fits):
        if fit is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(fit[1])
    tier_median_slope = {t: median(v) for t, v in tier_slopes.items()}
    global_slope = median([f[1] for f in fits if f is not None])
    global_resid = median([f[2] for f in fits if f is not None])

    rw_spread = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
    extrap = max(target_x - last_x, 0.0)

    for i, trophy in enumerate(trophies):
        fit = fits[i]
        rate = last_rates[i]
        if rate is None:
            rate = fallback_level(trophy["tier"])
        last_lg = logit(rate)
        if fit is None:
            med = last_lg
            spread = max(rw_spread, SPREAD_MIN)
        else:
            _, slope, resid, n, sxx = fit
            # Empirical-Bayes shrinkage of the slope toward the tier median.
            prior = tier_median_slope.get(trophy["tier"], global_slope)
            if sxx > 0 and resid > 0:
                se2 = (resid * resid) / sxx
                w = SLOPE_TAU * SLOPE_TAU / (SLOPE_TAU * SLOPE_TAU + se2)
                slope = w * slope + (1.0 - w) * prior
            elif sxx <= 0:
                slope = prior
            move = slope * extrap
            move = max(-MAX_MOVE, min(MAX_MOVE, move))
            med = last_lg + move
            # Spread: random walk + shrunk residual + extrapolation penalty.
            resid_shrunk = math.sqrt(
                (n * resid * resid + 5.0 * global_resid * global_resid) / (n + 5.0)
            )
            trend_unc = 0.5 * abs(move)
            spread = math.sqrt(
                rw_spread * rw_spread + resid_shrunk * resid_shrunk + trend_unc * trend_unc
            )
            spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))
        qs = []
        prev = None
        for z in Z:
            factor = SPREAD_DOWN_FACTOR if z < 0 else SPREAD_UP_FACTOR
            q = round4(min(100.0, max(0.0, inv_logit(med + z * spread * factor))))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
