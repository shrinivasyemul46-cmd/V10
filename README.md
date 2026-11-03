# 🎯 POSITIONAL & SWING BREAKOUT SCANNER — V10

Top industry groups → strong stocks that are coiled in a 6-24 month base, or breaking a 3-month / 6-month / 52-week
high with volume → fundamentals check → entry, stop, targets, quantity → backtest → journal.

## Files (all 4 in the GitHub repo root, exact names)
`app.py` · `engine.py` · `requirements.txt` · `README.md` — upload over the old ones, then **Reboot** the app.

## What is new in V10 (vs V9)
* **Trade horizon switch (top of sidebar):** *Positional (weeks-months)* = max stop 12%, 3R room to resistance, trail under
  EMA50, hold up to ~60 sessions. *Swing (3-10 days)* = max stop 8%, 2.5R room, trail under EMA20, hold ~15-20 sessions.
  It changes starting values and the exit plan; the Backtest tab follows the same horizon (trail EMA, holding days).
* **Volume check window (1-3 sessions):** 1 = breakout day alone. 2-3 = the AVERAGE volume ratio of the last 2-3 sessions must
  reach the threshold (breakout day still ≥ 1.5×). Today is never part of its own 20-day average (a true 7× day reads 7×).
* **Breakout level label:** each card says whether price broke a 3-month, 6-month or 52-week high (or a base ceiling).
* **Sector ranking** uses 60D / 20D median return + % of stocks above EMA50 **and EMA200**, with a minimum group size slider.
* **Plain-language verdict:** "N trade ideas today", or "No trade today" with the real reasons (regime, or the most common
  blockers among near-miss stocks).
* **Download** the day's picks as CSV.

## Everything from V9 is unchanged
6-24 month base breakout + coiled "ready" list, MFI/CCI/RSI momentum gates, transparent fundamental score (NA goes to a
manual-check list), market cap / price / circuit gates, world-India regime, reward:risk room to resistance, position
sizing, 3 entry × 2 exit backtest comparison, journal, simple mode.

## Honest limits
* Free end-of-day Yahoo data (shown as "price data as of"); fundamentals are patchy for small caps and are latest values.
* Backtest tests the chart rules only (no fundamentals, regime, sector or market-cap replay), uses today's index members,
  and has not yet been run on real NSE data by me. Score ≠ probability. Paper-trade first.
* Educational tool, not investment advice.
