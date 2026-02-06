import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA
# ===============================
FILE_PATH = "train.xlsx"   # change if needed
DATE_COL = "PRICES"        # your date column

df = pd.read_excel(FILE_PATH)

# Convert date column to datetime
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# Hour columns (Hour 01 ... Hour 24)
hour_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Hour ")]
if len(hour_cols) != 24:
    raise ValueError(f"Expected 24 hour columns, found {len(hour_cols)}: {hour_cols}")

# Ensure numeric
df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors="coerce")

# Calendar features
df["month"] = df[DATE_COL].dt.month
df["weekday"] = df[DATE_COL].dt.weekday  # Mon=0 ... Sun=6
df["day_type"] = df["weekday"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

hours = list(range(1, 25))  # x-axis

# ===============================
# Helper: add buy/sell windows
# ===============================
WINTER_MONTHS = {10, 11, 12, 1, 2}

def add_trade_windows(ax):
    """
    Adds shaded regions for buy/sell windows.
    - Buy: 2..7 (both seasons)
    - Sell winter: 10..12 and 18..21
    - Sell summer: 9..14
    """
    # Common buy window
    buy_window = [(2, 7)]
    # Winter vs summer sell windows
    winter_sell = [(10, 12), (18, 21)]
    summer_sell = [(9, 14)]

    # Shaded bands
    for (a, b) in buy_window:
        ax.axvspan(a, b, alpha=0.12, label="Buy window" if a == 2 else None)
    for (a, b) in summer_sell:
        ax.axvspan(a, b, alpha=0.12, label="Sell window (summer)")
    for (a, b) in winter_sell:
        ax.axvspan(a, b, alpha=0.12, label="Sell window (winter)" if a == 10 else None)

    # Make the band colors explicit via edge colors using lines (keeps default line colors untouched)
    # Draw bold boundary lines for readability
    def bold_bounds(windows, linestyle="-"):
        for (a, b) in windows:
            ax.axvline(a, linewidth=3, linestyle=linestyle)
            ax.axvline(b, linewidth=3, linestyle=linestyle)

    # Bold boundaries (optional but helps a lot)
    bold_bounds(buy_window, linestyle="--")
    bold_bounds(summer_sell, linestyle=":")
    bold_bounds(winter_sell, linestyle="-.")

# ===============================
# FIGURE 1:
# Typical 24h profile per month (12 lines)
# with CLEAR buy/sell windows
# ===============================

fig, ax = plt.subplots(figsize=(13, 6))

# ---------- BUY WINDOW (both seasons) ----------
ax.axvspan(2, 7,
           color="green", alpha=0.18,
           label="Buy window (all months)")
ax.axvline(2, color="green", linewidth=3, linestyle="--")
ax.axvline(7, color="green", linewidth=3, linestyle="--")

# ---------- SELL WINDOW (SUMMER) ----------
ax.axvspan(9, 14,
           color="tab:blue", alpha=0.18,
           label="Sell window (summer)")
ax.axvline(9, color="tab:blue", linewidth=3, linestyle=":")
ax.axvline(14, color="tab:blue", linewidth=3, linestyle=":")

# ---------- SELL WINDOWS (WINTER) ----------
ax.axvspan(10, 12,
           color="red", alpha=0.18,
           label="Sell window (winter – morning)")
ax.axvspan(18, 21,
           color="red", alpha=0.18,
           label="Sell window (winter – evening)")
ax.axvline(10, color="red", linewidth=3)
ax.axvline(12, color="red", linewidth=3)
ax.axvline(18, color="red", linewidth=3)
ax.axvline(21, color="red", linewidth=3)

# ---------- PLOT MONTH PROFILES ----------
for month in range(1, 13):
    month_profile = df[df["month"] == month][hour_cols].mean(axis=0)
    ax.plot(
        hours,
        month_profile.values,
        label=pd.to_datetime(str(month), format="%m").strftime("%B"),
        linewidth=1.8
    )

ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Price")
ax.set_title("Typical Day (24h Profile) by Month with Buy/Sell Windows")
ax.set_xticks(hours)
ax.grid(True, alpha=0.3)

# ---------- CLEAN LEGEND (no duplicates) ----------
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), ncol=3, fontsize=9)

plt.tight_layout()
plt.show()
