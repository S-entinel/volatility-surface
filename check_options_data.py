"""
Diagnostic: see exactly where option contracts drop out of the filters.

Uses only yfinance and pandas (none of the project code), so it shows what Yahoo
is really returning right now.

Usage (from the repo root, with the virtual environment active):
    python check_options_data.py            # SPY, minimum volume 10
    python check_options_data.py AAPL       # another ticker
    python check_options_data.py AAPL 0     # another ticker, minimum volume 0
"""
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
MIN_VOLUME = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MIN_STRIKE_PCT, MAX_STRIKE_PCT = 75, 125   # same defaults as the app
MIN_DAYS_TO_EXPIRY = 7                     # same as the app
MAX_EXPIRIES_TO_CHECK = 6

# ---------------------------------------------------------------- market clock
now_et = datetime.now(ZoneInfo("America/New_York"))
now_uk = datetime.now(ZoneInfo("Europe/London"))
clock = (now_et.hour, now_et.minute)
market_open = now_et.weekday() < 5 and (9, 30) <= clock < (16, 0)   # ignores public holidays
print(f"Now: {now_uk:%a %d %b %H:%M} UK  /  {now_et:%H:%M} New York")
print(f"US stock market open (9:30-16:00 New York, weekdays): {'YES' if market_open else 'NO'}")
print(f"Ticker: {SYMBOL} | min volume: {MIN_VOLUME} | strikes {MIN_STRIKE_PCT}%-{MAX_STRIKE_PCT}% of spot")

# ----------------------------------------------------------------- spot price
ticker = yf.Ticker(SYMBOL)
history = ticker.history(period="5d")
spot = float(history["Close"].iloc[-1])
print(f"Spot: {spot:.2f}  (last price bar: {history.index[-1]})")

# ---------------------------------------------------------------- expirations
expiries = []
for text in ticker.options:
    days = (datetime.strptime(text, "%Y-%m-%d").date() - date.today()).days
    if days >= MIN_DAYS_TO_EXPIRY:
        expiries.append((text, days))
print(f"Expiries at least {MIN_DAYS_TO_EXPIRY} days out: {len(expiries)} "
      f"(checking the first {min(len(expiries), MAX_EXPIRIES_TO_CHECK)})\n")

# ------------------------------------------------------------ filter counting
low, high = spot * MIN_STRIKE_PCT / 100, spot * MAX_STRIKE_PCT / 100
rows = []
sample = None

for text, days in expiries[:MAX_EXPIRIES_TO_CHECK]:
    chain = ticker.option_chain(text)
    for side, frame in (("call", chain.calls), ("put", chain.puts)):
        quoted = (frame["bid"] > 0) & (frame["ask"] > 0)
        in_range = frame["strike"].between(low, high)
        volume = frame["volume"].fillna(0) if "volume" in frame else pd.Series(0, index=frame.index)
        open_interest = (frame["openInterest"].fillna(0) if "openInterest" in frame
                         else pd.Series(0, index=frame.index))

        rows.append({
            "expiry": text, "days": days, "side": side,
            "total": len(frame),
            "quoted (bid&ask>0)": int(quoted.sum()),
            "+ in strike range": int((quoted & in_range).sum()),
            f"+ volume>={MIN_VOLUME}  (APP FILTER)": int((quoted & in_range & (volume >= MIN_VOLUME)).sum()),
            "alt: quoted+range+open interest>0": int((quoted & in_range & (open_interest > 0)).sum()),
        })

        if sample is None and side == "call":
            near_spot = frame.iloc[(frame["strike"] - spot).abs().argsort()[:6]].sort_values("strike")
            wanted = [c for c in ("strike", "bid", "ask", "volume", "openInterest", "lastTradeDate")
                      if c in near_spot.columns]
            sample = (text, near_spot[wanted])

table = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(table.to_string(index=False))

app_column = f"+ volume>={MIN_VOLUME}  (APP FILTER)"
calls_kept = int(table.loc[table["side"] == "call", app_column].sum())
puts_kept = int(table.loc[table["side"] == "put", app_column].sum())
print(f"\nCalls kept by the app's filters (what it uses today): {calls_kept}   <- app needs at least 10")
print(f"Puts kept by the same filters (not fetched by the app yet): {puts_kept}")

if sample is not None:
    print(f"\nSample of calls nearest the spot price, expiry {sample[0]}:")
    print(sample[1].to_string(index=False))