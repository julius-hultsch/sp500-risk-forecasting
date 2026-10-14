import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Load previously saved backtest results
bt = pd.read_csv(
    "data/backtest_results.csv",
    index_col="Date",
    parse_dates=True
)

alpha = 0.01


# ==========================================
# 1. Kupiec unconditional coverage test
# ==========================================

def kupiec_test(breaches, alpha=0.01):
    breaches = np.asarray(breaches, dtype=bool)
    n = len(breaches)
    x = int(breaches.sum())
    p_hat = x / n

    terms = 0.0

    if x > 0:
        terms += x * np.log(p_hat / alpha)

    if x < n:
        terms += (n - x) * np.log(
            (1 - p_hat) / (1 - alpha)
        )

    lr_stat = 2 * terms
    p_value = chi2.sf(lr_stat, df=1)

    return {
        "Observations": n,
        "Breaches": x,
        "Observed rate (%)": 100 * p_hat,
        "Expected breaches": n * alpha,
        "LR statistic": lr_stat,
        "p-value": p_value,
        "Reject at 5%": p_value < 0.05
    }


kupiec_results = pd.DataFrame({
    "Gaussian": kupiec_test(bt["Breach_Normal"], alpha),
    "Student-t": kupiec_test(bt["Breach_StudentT"], alpha)
}).T

print("\nKUPIEC TEST RESULTS")
print(kupiec_results.round(4))


# ==========================================
# 2. Plot actual losses and VaR forecasts
# ==========================================

fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(
    bt.index, 100 * bt["Loss"],
    color="grey", linewidth=0.7, alpha=0.55,
    label="Actual daily loss"
)

ax.plot(
    bt.index, 100 * bt["VaR_Normal"],
    linewidth=1.2, label="Gaussian 99% VaR"
)

ax.plot(
    bt.index, 100 * bt["VaR_StudentT"],
    linewidth=1.2, label="Student-t 99% VaR"
)

g = bt["Breach_Normal"].astype(bool)
t = bt["Breach_StudentT"].astype(bool)

ax.scatter(
    bt.index[g], 100 * bt.loc[g, "Loss"],
    marker="x", s=25, label="Gaussian breach"
)

ax.scatter(
    bt.index[t], 100 * bt.loc[t, "Loss"],
    marker="o", facecolors="none", s=35,
    label="Student-t breach"
)

ax.axvspan(
    pd.Timestamp("2008-09-01"),
    pd.Timestamp("2009-06-30"),
    color="grey", alpha=0.12,
    label="2008 crisis window"
)

ax.axvspan(
    pd.Timestamp("2020-02-15"),
    pd.Timestamp("2020-05-31"),
    color="orange", alpha=0.10,
    label="2020 COVID window"
)

ax.set_ylabel("Daily loss / VaR (%)")
ax.set_title("S&P 500: Actual Losses vs. Rolling 99% VaR")
ax.legend(loc="upper left", ncol=2)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ==========================================
# 3. Rolling breach rates
# ==========================================

rolling_window = 250

rolling_normal = (
    bt["Breach_Normal"].rolling(rolling_window).mean() * 100
)

rolling_student = (
    bt["Breach_StudentT"].rolling(rolling_window).mean() * 100
)

plt.figure(figsize=(15, 6))

plt.plot(
    rolling_normal.index, rolling_normal,
    label="Gaussian", linewidth=1.2
)

plt.plot(
    rolling_student.index, rolling_student,
    label="Student-t", linewidth=1.2
)

plt.axhline(
    1.0, linestyle="--", color="black",
    label="Target breach rate (1%)"
)

plt.axvspan(
    pd.Timestamp("2008-09-01"),
    pd.Timestamp("2009-06-30"),
    color="grey", alpha=0.12
)

plt.axvspan(
    pd.Timestamp("2020-02-15"),
    pd.Timestamp("2020-05-31"),
    color="orange", alpha=0.10
)

plt.ylabel("Breach rate over preceding 250 days (%)")
plt.title("Rolling VaR Breach Rates")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# ==========================================
# 4. Crisis-period comparison
# ==========================================

periods = {
    "2008 crisis": ("2008-09-01", "2009-06-30"),
    "2020 COVID crash": ("2020-02-15", "2020-05-31"),
    "Full backtest": (
        bt.index.min().strftime("%Y-%m-%d"),
        bt.index.max().strftime("%Y-%m-%d")
    )
}

rows = []

for period, (start, end) in periods.items():
    subset = bt.loc[start:end]

    if subset.empty:
        continue

    rows.append({
        "Period": period,
        "Days": len(subset),
        "Gaussian breaches": int(
            subset["Breach_Normal"].sum()
        ),
        "Gaussian rate (%)": (
            100 * subset["Breach_Normal"].mean()
        ),
        "Student-t breaches": int(
            subset["Breach_StudentT"].sum()
        ),
        "Student-t rate (%)": (
            100 * subset["Breach_StudentT"].mean()
        )
    })

crisis_table = pd.DataFrame(rows).set_index("Period")

print("\nCRISIS-PERIOD COMPARISON")
print(crisis_table.to_string(float_format=lambda x: f"{x:.2f}"))

print("\nSTUDENT-T PARAMETER DIAGNOSTICS")

print(bt["DF_StudentT"].describe())

print("\nLowest degrees of freedom:")
print(bt.nsmallest(10, "DF_StudentT")[
    ["DF_StudentT", "VaR_StudentT", "ES_StudentT"]
])

print("\nHighest Student-t VaR forecasts:")
print(bt.nlargest(10, "VaR_StudentT")[
    ["DF_StudentT", "VaR_StudentT", "ES_StudentT"]
])

cols = [
    "Loss",
    "VaR_Normal",
    "VaR_StudentT",
    "ES_StudentT",
    "DF_StudentT",
    "Breach_Normal",
    "Breach_StudentT"
]

print("\nLARGEST STUDENT-T VAR FORECASTS")

print(
    bt.nlargest(15, "VaR_StudentT")[cols]
    .to_string(float_format=lambda x: f"{x:.4f}")
)