
import numpy as np
import pandas as pd
from scipy.stats import chi2


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


# Load the existing Gaussian and Student-t results
df = pd.read_csv("data/backtest_results.csv")

# Load GARCH results and calculate VaR breaches
garch_df = pd.read_csv("data/garch_results.csv")

garch_df["Breach_GARCH"] = (
    garch_df["Loss"] > garch_df["VaR_GARCH"]
).astype(int)

# Keep only dates present in both files
df = df.merge(
    garch_df[["Date", "Breach_GARCH"]],
    on="Date",
    how="inner"
)

print("Common observations:", len(df))

models = {
    "Gaussian": "Breach_Normal",
    "Student-t": "Breach_StudentT",
    "GARCH-Student-t": "Breach_GARCH",
}

print("\nCONDITIONAL COVERAGE TEST")
print("Null hypothesis: correct 1% coverage and independent breaches")
print("-" * 72)

for model, column in models.items():
    breaches = df[column].dropna().astype(int).to_numpy()

    lr_uc = kupiec_test(breaches)
    lr_ind = christoffersen_independence_test(breaches)

    lr_cc = lr_uc + lr_ind
    p_value = chi2.sf(lr_cc, df=2)

    print(f"\n{model}")
    print(f"Observations: {len(breaches)}")
    print(f"Breach rate: {breaches.mean():.3%}")
    print(f"LR coverage: {lr_uc:.4f}")
    print(f"LR independence: {lr_ind:.4f}")
    print(f"LR conditional coverage: {lr_cc:.4f}")
    print(f"Conditional coverage p-value: {p_value:.6g}")
    print("Reject at 5%:", p_value < 0.05)
