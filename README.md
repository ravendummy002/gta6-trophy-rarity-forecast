# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ravendummy002/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ravendummy002/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates on Exophase: given the house's recording of every trophy's rate up to the case checkpoint, it predicts each of the 55 trophies' rate a week after launch (2026-11-26 00:00 UTC) as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

How it works (`main.py`, Python 3.13 standard library only, deterministic, offline):

- **Moving history** (the final case): deduplicates Exophase's repeated page states, then fits each trophy a least-squares line of logit rate against ln(1 + hours since release) over the last 24 hours of reads. Slopes are shrunk toward their tier's median by empirical-Bayes precision weighting. The median anchors on the last observed rate; the spread combines a random-walk term, the shrunk fit residual, and an extrapolation penalty, mildly asymmetric (wider below the median, since arriving players dilute every rate).
- **Flat history** (the practice cases): an unchanged page means a refresh is due, and the launch-night record shows a refresh dilutes every non-universal trophy by roughly the player-growth factor — so the median shifts down by 1.6 · (1 − √rate) logit units with a wide asymmetric spread.

Local verification (scorer replicated from the rules and checked against the worked example): on the practice cases, against a dilution-aware proxy answer reconstructed from the published launch-night aggregates, this forecaster scores a pinball loss of 0.14, versus 1.39 for the house persistence baseline, 1.48 for last-value-carried-forward with zero spread, and 2.86 for a global-mean forecast.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<cases-dir> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 /bot/main.py` with the case in `/input` and expects `/output/<id>.out`. Two runs on the same case write identical bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=r16q3s2q&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
