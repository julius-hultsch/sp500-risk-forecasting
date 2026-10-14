import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
# ==========================================
# 1. Download S&P 500 data
# ==========================================

prices = yf.download(
    "^GSPC",
    start="2000-01-01",
    end="2026-01-01",
    auto_adjust=True,
    progress=False
)

# Handle the column structure returned by different yfinance versions
close = prices["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

close = close.dropna().rename("Price")

# Daily log returns
returns = np.log(close / close.shift(1)).dropna()
returns.name = "Return"

print(returns.head())
print(returns.describe())
print(f"Observations: {len(returns):,}")

# ==========================================
# 2. Fit Gaussian and Student-t distributions
# ==========================================
x = returns.to_numpy()

# Gaussian maximum-likelihood estimates
mu_norm, sigma_norm = stats.norm.fit(x)

# Student-t maximum-likelihood estimates
df_t, loc_t, scale_t = stats.t.fit(x)

print("\nGaussian")
print(f"Mean: {mu_norm:.6f}")
print(f"Standard deviation: {sigma_norm:.6f}")

print("\nStudent-t")
print(f"Degrees of freedom: {df_t:.3f}")
print(f"Location: {loc_t:.6f}")
print(f"Scale: {scale_t:.6f}")


### Visualising the tails
grid = np.linspace(
    returns.quantile(0.001),
    returns.quantile(0.999),
    1000
)

plt.figure(figsize=(10, 6))

sns.histplot(
    returns,
    bins=100,
    stat="density",
    color="grey",
    alpha=0.35,
    label="Observed returns"
)

plt.plot(
    grid,
    stats.norm.pdf(grid, mu_norm, sigma_norm),
    label="Gaussian"
)

plt.plot(
    grid,
    stats.t.pdf(grid, df_t, loc_t, scale_t),
    label="Student-t"
)

plt.xlim(-0.08, 0.08)
plt.xlabel("Daily log return")
plt.ylabel("Density")
plt.title("Distribution of S&P 500 Daily Returns")
plt.legend()
plt.show()
PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
plt.savefig(
    FIGURES_DIR / "crisis_breach_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# ==========================================
# 3. Rolling VaR/ES backtest
# ==========================================
alpha = 0.01
window = 1000
refit_every = 5

r = returns.to_numpy()
dates = returns.index

results = []

# Cached model parameters
normal_params = None
student_params = None

for i in range(window, len(r)):

    # Refit using past observations only
    if normal_params is None or (i - window) % refit_every == 0:
        train = r[i - window:i]

        normal_params = stats.norm.fit(train)
        student_params = stats.t.fit(train)

    mu, sigma = normal_params
    df, loc, scale = student_params

    # Gaussian VaR and ES, expressed as positive losses
    z = stats.norm.ppf(alpha)

    var_normal = -(mu + sigma * z)
    es_normal = -mu + sigma * stats.norm.pdf(z) / alpha

    # Student-t VaR and ES
    q_t = stats.t.ppf(alpha, df)

    var_t = -(loc + scale * q_t)

    if df > 1:
        es_t = (
            -loc
            + scale
            * stats.t.pdf(q_t, df)
            * (df + q_t**2)
            / ((df - 1) * alpha)
        )
    else:
        es_t = np.inf

    actual_return = r[i]
    actual_loss = -actual_return

    results.append({
        "Date": dates[i],
        "Return": actual_return,
        "Loss": actual_loss,
        "VaR_Normal": var_normal,
        "ES_Normal": es_normal,
        "VaR_StudentT": var_t,
        "ES_StudentT": es_t,
        "DF_StudentT": df
    })

bt = pd.DataFrame(results).set_index("Date")

bt["Breach_Normal"] = bt["Loss"] > bt["VaR_Normal"]
bt["Breach_StudentT"] = bt["Loss"] > bt["VaR_StudentT"]

print(bt.head())
print(f"\nBacktest observations: {len(bt):,}")

### Comparing breach rates
summary = pd.DataFrame({
    "Model": ["Gaussian", "Student-t"],
    "Mean VaR (%)": [
        100 * bt["VaR_Normal"].mean(),
        100 * bt["VaR_StudentT"].mean()
    ],
    "Mean ES (%)": [
        100 * bt["ES_Normal"].mean(),
        100 * bt["ES_StudentT"].mean()
    ],
    "Breach count": [
        bt["Breach_Normal"].sum(),
        bt["Breach_StudentT"].sum()
    ],
    "Breach rate (%)": [
        100 * bt["Breach_Normal"].mean(),
        100 * bt["Breach_StudentT"].mean()
    ]
})

#print(summary.round(3).to_string(index=False))
# saving the backtest results as a csv
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

bt.to_csv(DATA_DIR / "backtest_results.csv")
print("\nBacktest results saved.")