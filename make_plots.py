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
# FIGURE 1:
# Typical 24h profile per month (12 lines)
# ===============================
plt.figure(figsize=(12, 6))

for month in range(1, 13):
    month_profile = df[df["month"] == month][hour_cols].mean(axis=0)  # mean over days, per hour
    plt.plot(hours, month_profile.values,
             label=pd.to_datetime(str(month), format="%m").strftime("%B"))

plt.xlabel("Hour of Day")
plt.ylabel("Average Price")
plt.title("Typical Day (24h Profile) by Month")
plt.xticks(hours)
plt.grid(True)
plt.legend(ncol=3)
plt.tight_layout()
plt.show()

# ===============================
# FIGURE 2:
# Typical weekday vs weekend 24h profile (2 lines)
# ===============================
weekday_profile = df[df["day_type"] == "Weekday"][hour_cols].mean(axis=0)
weekend_profile = df[df["day_type"] == "Weekend"][hour_cols].mean(axis=0)

plt.figure(figsize=(12, 6))
plt.plot(hours, weekday_profile.values, label="Weekday (avg)")
plt.plot(hours, weekend_profile.values, label="Weekend (avg)")

plt.xlabel("Hour of Day")
plt.ylabel("Average Price")
plt.title("Typical Day (24h Profile): Weekday vs Weekend")
plt.xticks(hours)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
