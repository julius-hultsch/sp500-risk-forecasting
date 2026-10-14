import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2

# --------------------------------------------------
# 1. Backtesting statistical test functions
# --------------------------------------------------
def kupiec_test(breaches, target=0.01):
    x = np.asarray(breaches, dtype=int)
    n = len(x)
    k = int(x.sum())
    p_hat = k / n

    def log_likelihood(p):
        if p == 0:
            return 0.0 if k == 0 else -np.inf
        if p == 1:
            return 0.0 if k == n else -np.inf
        return k * np.log(p) + (n - k) * np.log1p(-p)

    lr_uc = 2 * (log_likelihood(p_hat) - log_likelihood(target))
    return max(0.0, lr_uc)


def christoffersen_independence_test(breaches):
    x = np.asarray(breaches, dtype=int)
    previous, current = x[:-1], x[1:]

    n00 = np.sum((previous == 0) & (current == 0))
    n01 = np.sum((previous == 0) & (current == 1))
    n10 = np.sum((previous == 1) & (current == 0))
    n11 = np.sum((previous == 1) & (current == 1))

    def contribution(successes, failures):
        total = successes + failures
        if total == 0:
            return 0.0
        p = successes / total
        if p == 0 or p == 1:
            return 0.0
        return (
            successes * np.log(p)
            + failures * np.log1p(-p)
        )

    total_breaches = n01 + n11
    total_transitions = n00 + n01 + n10 + n11
    p = total_breaches / total_transitions

    ll_independent = (
        (total_breaches * np.log(p)
         if p > 0 else 0.0)
        + ((total_transitions - total_breaches) * np.log1p(-p)
           if p < 1 else 0.0)
    )

    ll_dependent = (
        contribution(n01, n00)
        + contribution(n11, n10)
    )

    lr_ind = max(0.0, 2 * (ll_dependent - ll_independent))
    return lr_ind

# --------------------------------------------------
# 2. Set up directories
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 3. Load and align the results
# --------------------------------------------------

baseline = pd.read_csv(
    "data/backtest_results.csv", parse_dates=["Date"]
)

garch = pd.read_csv(
    "data/garch_results.csv", parse_dates=["Date"]
)

df = baseline.merge(
    garch[
        ["Date", "VaR_GARCH", "ES_GARCH", "GARCH_DF"]
    ],
    on="Date",
    how="inner",
    validate="one_to_one"
)

# Ensure chronological ordering
df = df.sort_values("Date").reset_index(drop=True)

# Recalculate breaches consistently for all models
df["Breach_Normal"] = (
    df["Loss"] > df["VaR_Normal"]
).astype(int)

df["Breach_StudentT"] = (
    df["Loss"] > df["VaR_StudentT"]
).astype(int)

df["Breach_GARCH"] = (
    df["Loss"] > df["VaR_GARCH"]
).astype(int)

models = {
    "Gaussian": ("VaR_Normal", "ES_Normal", "Breach_Normal"),
    "Student-t": ("VaR_StudentT", "ES_StudentT", "Breach_StudentT"),
    "GARCH-Student-t": ("VaR_GARCH", "ES_GARCH", "Breach_GARCH"),
}

print("Common observations:", len(df))


# --------------------------------------------------
# 4. Final overall results table
# --------------------------------------------------

summary = []

for model, (var_col, es_col, breach_col) in models.items():
    summary.append({
        "Model": model,
        "Observations": len(df),
        "Mean_VaR_pct": df[var_col].mean() * 100,
        "Mean_ES_pct": df[es_col].mean() * 100,
        "Breach_Count": int(df[breach_col].sum()),
        "Breach_Rate_pct": df[breach_col].mean() * 100,
        "Target_Rate_pct": 1.0,
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(DATA_DIR / "final_model_comparison.csv", index=False)

print("\nFINAL MODEL COMPARISON")
print(summary_df.round(3).to_string(index=False))

# --------------------------------------------------
# 5. Conditional-coverage backtesting
# --------------------------------------------------

coverage_results = []

print("\nCONDITIONAL COVERAGE TEST")
print("Null hypothesis: correct 1% coverage and independent breaches")
print("-" * 72)

for model, (_, _, breach_col) in models.items():
    breaches = df[breach_col].dropna().astype(int).to_numpy()

    lr_uc = kupiec_test(breaches, target=0.01)
    lr_ind = christoffersen_independence_test(breaches)

    lr_cc = lr_uc + lr_ind
    p_value = chi2.sf(lr_cc, df=2)

    coverage_results.append({
        "Model": model,
        "Observations": len(breaches),
        "Breach_Rate_pct": breaches.mean() * 100,
        "LR_Coverage": lr_uc,
        "LR_Independence": lr_ind,
        "LR_Conditional_Coverage": lr_cc,
        "Conditional_Coverage_p_value": p_value,
        "Reject_at_5pct": p_value < 0.05,
    })

    print(f"\n{model}")
    print(f"Observations: {len(breaches)}")
    print(f"Breach rate: {breaches.mean():.3%}")
    print(f"LR coverage: {lr_uc:.4f}")
    print(f"LR independence: {lr_ind:.4f}")
    print(f"LR conditional coverage: {lr_cc:.4f}")
    print(f"Conditional coverage p-value: {p_value:.6g}")
    print("Reject at 5%:", p_value < 0.05)

coverage_df = pd.DataFrame(coverage_results)
coverage_df.to_csv(DATA_DIR / "conditional_coverage_results.csv", index=False)


# --------------------------------------------------
# 6. Chart: VaR forecasts and realized losses
# --------------------------------------------------

# Show 2020 in detail to avoid an unreadable full-sample plot
start = "2020-02-01"
end = "2020-05-31"

period = df[
    (df["Date"] >= start) &
    (df["Date"] <= end)
].copy()

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    period["Date"], period["Loss"] * 100,
    color="black", linewidth=0.8,
    label="Realized loss"
)

ax.plot(
    period["Date"], period["VaR_Normal"] * 100,
    label="Gaussian VaR", linewidth=1.2
)

ax.plot(
    period["Date"], period["VaR_StudentT"] * 100,
    label="Student-t VaR", linewidth=1.2
)

ax.plot(
    period["Date"], period["VaR_GARCH"] * 100,
    label="GARCH-Student-t VaR", linewidth=1.2
)

ax.axhline(0, color="grey", linewidth=0.7)
ax.set_title("S&P 500: 99% VaR Forecasts During the COVID Crash")
ax.set_ylabel("Daily loss / VaR (%)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "var_covid_crash.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# --------------------------------------------------
# 7. Chart: rolling 250-observation breach rates
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 6))

for model, (_, _, breach_col) in models.items():
    rolling_rate = (
        df[breach_col]
        .rolling(250, min_periods=250)
        .mean() * 100
    )

    ax.plot(
        df["Date"], rolling_rate,
        label=model, linewidth=1.1
    )

# Highlight breaches for each model
breach_colors = {
    "VaR_Normal": "blue",
    "VaR_StudentT": "darkorange",
    "VaR_GARCH": "green",
}

for var_col, color in breach_colors.items():
    breached = period["Loss"] > period[var_col]

    ax.scatter(
        period.loc[breached, "Date"],
        period.loc[breached, "Loss"] * 100,
        color=color,
        marker="o",
        s=35,
        edgecolors="black",
        linewidths=0.4,
        zorder=5,
        label=f"{var_col} breach"
    )

ax.axhline(
    1.0, color="black", linestyle="--",
    linewidth=1.2, label="1% target"
)

ax.set_title("Rolling 250-Day VaR Breach Rates")
ax.set_ylabel("Breach rate (%)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "rolling_breach_rates.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# --------------------------------------------------
# 8t. Crisis-period comparison
# --------------------------------------------------

crisis_periods = {
    "2008 Financial Crisis": ("2008-09-01", "2009-06-30"),
    "2020 COVID Crash": ("2020-02-15", "2020-05-31"),
}

crisis_results = []

for crisis, (start, end) in crisis_periods.items():
    subset = df[
        (df["Date"] >= start) &
        (df["Date"] <= end)
    ]

    for model, (_, _, breach_col) in models.items():
        crisis_results.append({
            "Period": crisis,
            "Model": model,
            "Observations": len(subset),
            "Breach_Count": int(subset[breach_col].sum()),
            "Breach_Rate_pct": subset[breach_col].mean() * 100,
        })

crisis_df = pd.DataFrame(crisis_results)
crisis_df.to_csv(DATA_DIR / "crisis_comparison.csv", index=False)

print("\nCRISIS-PERIOD COMPARISON")
print(crisis_df.round(3).to_string(index=False))

# Plot crisis breach rates
pivot = crisis_df.pivot(
    index="Period",
    columns="Model",
    values="Breach_Rate_pct"
)

ax = pivot.plot(
    kind="bar", figsize=(10, 6), rot=0
)

ax.axhline(
    1.0, color="black", linestyle="--",
    linewidth=1.2, label="1% target"
)

ax.set_title("VaR Breach Rates During Market Crises")
ax.set_ylabel("Breach rate (%)")
ax.set_xlabel("")
ax.legend(title="Model")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "crisis_breach_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\nSaved:")
print("- final_model_comparison.csv")
print("- crisis_comparison.csv")
print("- var_covid_crash.png")
print("- rolling_breach_rates.png")
print("- crisis_breach_comparison.png")
