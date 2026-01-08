# Exploratory Data Analysis (EDA) for train.xlsx
# - Loads the file
# - Cleans / reshapes hourly columns
# - Creates lots of graphs
# - Includes a dedicated plot for overall average price + standard deviation

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1) Load data
# ----------------------------
FILE_PATH = "DATA/train.xlsx"  # <- change if needed
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_excel(FILE_PATH)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nHead:\n", df.head())

# ----------------------------
# 2) Identify date column + hour columns
# ----------------------------
# Your file appears to have:
# - a date column called "PRICES"
# - hourly columns like "Hour 01" ... "Hour 24"
date_col_candidates = [c for c in df.columns if str(c).strip().lower() in ["prices", "date", "day", "datetime"]]
date_col = date_col_candidates[0] if date_col_candidates else df.columns[0]

hour_cols = [c for c in df.columns if re.match(r"(?i)^hour\s*\d{1,2}$", str(c).strip())]
# If the file is exactly "Hour 01".."Hour 24" this should work.
if not hour_cols:
    # fallback: anything containing "hour"
    hour_cols = [c for c in df.columns if "hour" in str(c).lower()]

print("\nDetected date column:", date_col)
print("Detected hour columns:", hour_cols[:5], "...", hour_cols[-5:])
assert len(hour_cols) > 0, "No hour columns detected."

# ----------------------------
# 3) Basic cleaning
# ----------------------------
data = df.copy()

# Parse date column
data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
bad_dates = data[date_col].isna().sum()
if bad_dates:
    print(f"\nWARNING: {bad_dates} rows have unparsed dates and will be dropped.")
data = data.dropna(subset=[date_col]).sort_values(date_col)

# Coerce hour columns to numeric
for c in hour_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

# Missing values report
print("\nMissing values per column:")
print(data[[date_col] + hour_cols].isna().sum())

# Set datetime index
data = data.set_index(date_col)

# ----------------------------
# 4) Helpful derived series
# ----------------------------
# Daily average across all hours
daily_avg = data[hour_cols].mean(axis=1)

# Daily std across hours (within-day volatility across hours)
daily_std_within_day = data[hour_cols].std(axis=1)

# Overall (all values) mean & std
all_values = data[hour_cols].to_numpy().ravel()
all_values = all_values[~np.isnan(all_values)]
overall_mean = float(np.mean(all_values))
overall_std = float(np.std(all_values, ddof=1))

print("\nOverall mean price:", overall_mean)
print("Overall std (sample):", overall_std)

# Long format (date, hour, price) for several plots
long = (
    data[hour_cols]
    .reset_index()
    .melt(id_vars=[date_col], var_name="Hour", value_name="Price")
)

# Extract hour number for sorting (Hour 01 -> 1)
def hour_to_int(s):
    m = re.search(r"(\d{1,2})", str(s))
    return int(m.group(1)) if m else np.nan

long["HourNum"] = long["Hour"].apply(hour_to_int)
long = long.dropna(subset=["HourNum"])
long["HourNum"] = long["HourNum"].astype(int)

# ----------------------------
# 5) Plot helpers
# ----------------------------
def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print("Saved:", path)

# ----------------------------
# 6) Graphs
# ----------------------------

# (A) Time series: daily average price
plt.figure(figsize=(12, 4))
plt.plot(daily_avg.index, daily_avg.values)
plt.title("Daily Average Price (mean of Hour 01..Hour 24)")
plt.xlabel("Date")
plt.ylabel("Average Price")
savefig("01_daily_average_timeseries.png")
plt.show()

# (B) Time series: within-day hourly std (how spread out hours are per day)
plt.figure(figsize=(12, 4))
plt.plot(daily_std_within_day.index, daily_std_within_day.values)
plt.title("Within-Day Price Variability (std across 24 hours)")
plt.xlabel("Date")
plt.ylabel("Std across hours")
savefig("02_within_day_std_timeseries.png")
plt.show()

# (C) Distribution: all hourly prices combined (histogram)
plt.figure(figsize=(8, 4))
plt.hist(all_values, bins=60)
plt.title("Distribution of All Hourly Prices (All Days x 24 Hours)")
plt.xlabel("Price")
plt.ylabel("Count")
savefig("03_all_values_hist.png")
plt.show()

# (D) Dedicated: Overall average price + std (with mean ± 1 std band)
#     This is the "average price and the std of it" plot you asked for.
plt.figure(figsize=(10, 4))
plt.hist(all_values, bins=60, alpha=0.9)
plt.axvline(overall_mean, linewidth=2, label=f"Mean = {overall_mean:.2f}")
plt.axvline(overall_mean - overall_std, linestyle="--", linewidth=2, label=f"Mean - 1 std = {(overall_mean-overall_std):.2f}")
plt.axvline(overall_mean + overall_std, linestyle="--", linewidth=2, label=f"Mean + 1 std = {(overall_mean+overall_std):.2f}")
plt.title("All Hourly Prices with Overall Mean ± 1 Std")
plt.xlabel("Price")
plt.ylabel("Count")
plt.legend()
savefig("04_mean_std_overlay.png")
plt.show()

# (E) Boxplots by hour (how price varies by hour-of-day across all days)
# Build data in hour order 1..24
hour_order = sorted(long["HourNum"].unique())
box_data = [long.loc[long["HourNum"] == h, "Price"].dropna().values for h in hour_order]

plt.figure(figsize=(14, 5))
plt.boxplot(box_data, showfliers=False)
plt.title("Price by Hour of Day (Boxplots across all dates)")
plt.xlabel("Hour of Day")
plt.ylabel("Price")
plt.xticks(ticks=np.arange(1, len(hour_order) + 1), labels=[str(h).zfill(2) for h in hour_order])
savefig("05_boxplot_by_hour.png")
plt.show()

# (F) Hourly means across the day (average daily shape)
hourly_mean = long.groupby("HourNum")["Price"].mean()
hourly_std = long.groupby("HourNum")["Price"].std()

plt.figure(figsize=(10, 4))
plt.plot(hourly_mean.index, hourly_mean.values, marker="o")
plt.title("Average Price by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Price")
savefig("06_hourly_mean_curve.png")
plt.show()

# (G) Hourly mean with error bars (std across days) — another "mean + variability" view
plt.figure(figsize=(10, 4))
plt.errorbar(hourly_mean.index, hourly_mean.values, yerr=hourly_std.values, fmt="o-", capsize=3)
plt.title("Average Price by Hour (Error bars = std across days)")
plt.xlabel("Hour of Day")
plt.ylabel("Price")
savefig("07_hourly_mean_with_std_errorbars.png")
plt.show()

# (H) Heatmap-like image: dates (y) vs hour (x)
# Create matrix (rows: dates, cols: hour)
pivot = long.pivot_table(index=date_col, columns="HourNum", values="Price", aggfunc="mean").sort_index()

plt.figure(figsize=(12, 6))
plt.imshow(pivot.values, aspect="auto")
plt.title("Heatmap: Price by Date (rows) and Hour (columns)")
plt.xlabel("Hour of Day")
plt.ylabel("Date index (sorted)")
plt.xticks(ticks=np.arange(len(pivot.columns)), labels=[str(h).zfill(2) for h in pivot.columns], rotation=0)
plt.colorbar(label="Price")
savefig("08_heatmap_date_vs_hour.png")
plt.show()

# (I) Correlation between hours
corr = data[hour_cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr.values, aspect="equal")
plt.title("Correlation Matrix Between Hours")
plt.xticks(ticks=np.arange(len(hour_cols)), labels=[hour_to_int(h) for h in hour_cols], rotation=90)
plt.yticks(ticks=np.arange(len(hour_cols)), labels=[hour_to_int(h) for h in hour_cols])
plt.colorbar(label="Correlation")
savefig("09_hourly_correlation_matrix.png")
plt.show()

# (J) Rolling mean of daily average (trend)
rolling_window = 30  # days
roll = daily_avg.rolling(rolling_window).mean()

plt.figure(figsize=(12, 4))
plt.plot(daily_avg.index, daily_avg.values, alpha=0.35, label="Daily avg")
plt.plot(roll.index, roll.values, linewidth=2, label=f"{rolling_window}-day rolling mean")
plt.title("Daily Average Price with Rolling Mean")
plt.xlabel("Date")
plt.ylabel("Average Price")
plt.legend()
savefig("10_rolling_mean_daily_avg.png")
plt.show()

# (K) Scatter: daily average vs within-day hourly std
plt.figure(figsize=(6, 5))
plt.scatter(daily_avg.values, daily_std_within_day.values, alpha=0.6)
plt.title("Daily Average vs Within-Day Std (across hours)")
plt.xlabel("Daily average price")
plt.ylabel("Std across hours (within-day)")
savefig("11_scatter_daily_avg_vs_within_day_std.png")
plt.show()

print("\nDone. All figures saved to:", os.path.abspath(OUT_DIR))
