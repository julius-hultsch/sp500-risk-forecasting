import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
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

close = prices["Close"]

if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

returns = np.log(close / close.shift(1)).dropna()

# Convert returns to percentage units for numerical stability
r = returns.to_numpy() * 100
dates = returns.index


# ==========================================
# 2. Set up the rolling backtest
# ==========================================

alpha = 0.01
window = 1000
refit_every = 5

results = []

params = None
last_variance = None
last_residual = None

for i in range(window, len(r)):

    # Refit the GARCH model every five trading days
    if params is None or (i - window) % refit_every == 0:

        train = r[i - window:i]

        model = arch_model(
            train,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="StudentsT",
            rescale=False
        )

        fitted = model.fit(disp="off")

        params = fitted.params

        mu = params["mu"]
        omega = params["omega"]
        a1 = params["alpha[1]"]
        b1 = params["beta[1]"]
        nu = params["nu"]

        # Final in-sample conditional variance and residual
        last_variance = (
            fitted.conditional_volatility[-1] ** 2
        )

        last_residual = train[-1] - mu

    # One-step-ahead conditional variance forecast
    variance_forecast = (
        omega
        + a1 * last_residual**2
        + b1 * last_variance
    )

    sigma_forecast = np.sqrt(variance_forecast)

    # Diagnostic: inspect GARCH estimates during March 2020
    if dates[i].year == 2020 and dates[i].month == 3:
        print(
            dates[i],
            "sigma:", round(sigma_forecast, 3),
            "omega:", round(omega, 6),
            "alpha:", round(a1, 4),
            "beta:", round(b1, 4),
            "nu:", round(nu, 3)
        )

    # Standardised Student-t quantile
    q = stats.t.ppf(alpha, df=nu)
    standardisation = np.sqrt((nu - 2) / nu)
    q_standardised = q * standardisation

    # VaR and ES in percentage units
    var_garch = -(mu + sigma_forecast * q_standardised)

    es_garch = (
        -mu
        + sigma_forecast
        * standardisation
        * stats.t.pdf(q, df=nu)
        * (nu + q**2)
        / ((nu - 1) * alpha)
    )

    actual_return = r[i]
    actual_loss = -actual_return

    results.append({
        "Date": dates[i],
        "Return": actual_return / 100,
        "Loss": actual_loss / 100,
        "VaR_GARCH": var_garch / 100,
        "ES_GARCH": es_garch / 100,
        "GARCH_DF": nu
    })

    # Update volatility state using the observed return.
    # These observations are used only after their returns occur.
    last_residual = actual_return - mu
    last_variance = variance_forecast

    if (i - window + 1) % 250 == 0:
        print(
            f"Processed {i - window + 1:,} observations"
        )


# ==========================================
# 3. Save results
# ==========================================

garch_bt = pd.DataFrame(results).set_index("Date")

garch_bt["Breach_GARCH"] = (
    garch_bt["Loss"] > garch_bt["VaR_GARCH"]
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

garch_bt.to_csv(DATA_DIR / "garch_results.csv")

print("\nGARCH backtest complete!")
print(f"Observations: {len(garch_bt):,}")


# ==========================================
# 4. Compare with existing baseline models
# ==========================================

baseline = pd.read_csv(
    "data/backtest_results.csv",
    index_col="Date",
    parse_dates=True
)

# Keep only dates available in both experiments
comparison = baseline.join(
    garch_bt[
        ["VaR_GARCH", "ES_GARCH", "Breach_GARCH"]
    ],
    how="inner"
)

summary = pd.DataFrame({
    "Gaussian": [
        comparison["VaR_Normal"].mean() * 100,
        comparison["ES_Normal"].mean() * 100,
        comparison["Breach_Normal"].mean() * 100
    ],
    "Student-t": [
        comparison["VaR_StudentT"].mean() * 100,
        comparison["ES_StudentT"].mean() * 100,
        comparison["Breach_StudentT"].mean() * 100
    ],
    "GARCH-Student-t": [
        comparison["VaR_GARCH"].mean() * 100,
        comparison["ES_GARCH"].mean() * 100,
        comparison["Breach_GARCH"].mean() * 100
    ]
}, index=[
    "Mean VaR (%)",
    "Mean ES (%)",
    "Breach rate (%)"
])

print("\nMODEL COMPARISON")
print(summary.round(3))

print("\nBREACH COUNTS")
print(
    comparison[
        [
            "Breach_Normal",
            "Breach_StudentT",
            "Breach_GARCH"
        ]
    ].sum()
)

# Plot conditional VaR forecasts
plt.figure(figsize=(14, 6))

plt.plot(
    comparison.index,
    comparison["VaR_Normal"] * 100,
    label="Gaussian",
    linewidth=1
)

plt.plot(
    comparison.index,
    comparison["VaR_StudentT"] * 100,
    label="Student-t",
    linewidth=1
)

plt.plot(
    comparison.index,
    comparison["VaR_GARCH"] * 100,
    label="GARCH-Student-t",
    linewidth=1
)

plt.ylabel("99% VaR (%)")
plt.title("Comparison of Rolling VaR Forecasts")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

print("Reached GARCH diagnostics")

# Inspect the largest GARCH VaR forecasts
print("\nLargest GARCH VaR forecasts:")

largest = garch_bt.nlargest(10, "VaR_GARCH")
print(largest.to_string(index=True))

print("\nGARCH VaR summary:")
print(garch_bt["VaR_GARCH"].describe(
    percentiles=[0.50, 0.90, 0.95, 0.99]
))

print("\n--- GARCH VALIDATION ---")

print("Return range:", garch_bt["Return"].min(),
      "to", garch_bt["Return"].max())

print("VaR range:", garch_bt["VaR_GARCH"].min(),
      "to", garch_bt["VaR_GARCH"].max())

print("ES range:", garch_bt["ES_GARCH"].min(),
      "to", garch_bt["ES_GARCH"].max())

print("Minimum degrees of freedom:",
      garch_bt["GARCH_DF"].min())

# Basic consistency checks
print("Any missing VaR values:",
      garch_bt["VaR_GARCH"].isna().any())

print("Any infinite VaR values:",
      (~garch_bt["VaR_GARCH"].map(
          lambda x: abs(x) < float("inf")
      )).any())

print("Any ES below VaR:",
      (garch_bt["ES_GARCH"] < garch_bt["VaR_GARCH"]).any())