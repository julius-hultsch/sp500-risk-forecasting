
import numpy as np
import pandas as pd
from scipy.stats import chi2


def christoffersen_independence_test(breaches):
    """
    Christoffersen (1998) independence test for VaR breaches.
    breaches: sequence of 0/1 values or Boolean indicators.
    """
    x = np.asarray(breaches, dtype=int)
    x = x[~pd.isna(x)]

    if len(x) < 2:
        raise ValueError("At least two breach observations are required.")

    # Count transitions between consecutive observations
    previous = x[:-1]
    current = x[1:]

    n00 = np.sum((previous == 0) & (current == 0))
    n01 = np.sum((previous == 0) & (current == 1))
    n10 = np.sum((previous == 1) & (current == 0))
    n11 = np.sum((previous == 1) & (current == 1))

    # Probability of a breach under the independent model
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total

    # Transition-specific breach probabilities
    denom_01 = n00 + n01
    denom_11 = n10 + n11

    pi01 = n01 / denom_01 if denom_01 else 0.0
    pi11 = n11 / denom_11 if denom_11 else 0.0

    def log_bernoulli_likelihood(successes, failures, p):
        if p <= 0:
            return 0.0 if successes == 0 else -np.inf
        if p >= 1:
            return 0.0 if failures == 0 else -np.inf

        return successes * np.log(p) + failures * np.log1p(-p)

    log_l0 = (
        (n00 + n10) * np.log1p(-pi)
        + (n01 + n11) * np.log(pi)
        if 0 < pi < 1
        else 0.0
    )

    log_l1 = (
        log_bernoulli_likelihood(n01, n00, pi01)
        + log_bernoulli_likelihood(n11, n10, pi11)
    )

    lr_ind = max(0.0, -2 * (log_l0 - log_l1))
    p_value = chi2.sf(lr_ind, df=1)

    return {
        "n00": int(n00),
        "n01": int(n01),
        "n10": int(n10),
        "n11": int(n11),
        "LR_ind": lr_ind,
        "p_value": p_value,
        "reject_independence_at_5pct": p_value < 0.05,
    }


# Load existing backtest results
df = pd.read_csv("data/backtest_results.csv")

for model, column in [
    ("Gaussian", "Breach_Normal"),
    ("Student-t", "Breach_StudentT"),
]:
    result = christoffersen_independence_test(df[column])

    print(f"\n{model}")
    print(f"VaR breaches: {int(df[column].sum())}")
    print(f"Independence test statistic: {result['LR_ind']:.4f}")
    print(f"P-value: {result['p_value']:.6f}")
    print(
        "Reject independence at 5%:",
        result["reject_independence_at_5pct"]
    )
