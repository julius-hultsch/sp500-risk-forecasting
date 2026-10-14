# Final Analysis: Forecasting Extreme Downside Risk in the S&P 500

## 1. Research Objective

The objective of this project was to evaluate whether heavy-tailed return 
distributions and time-varying volatility improve the forecasting of extreme 
downside risk in equity markets.

The analysis compared a Gaussian benchmark, a Student-t return model, and a 
GARCH(1,1) model with Student-t innovations. Each model produced one-day-ahead 
forecasts of 99% Value at Risk (VaR) and Expected Shortfall (ES).

The evaluation focused on three dimensions: the frequency of VaR breaches, the 
temporal dependence of breaches, and the models' behavior during major market crises.

## 2. Full-Sample Results

The models were evaluated over 5,538 common out-of-sample observations.

| Metric | Gaussian | Student-t | GARCH-Student-t |
|---|---:|---:|---:|
| Mean VaR | 2.703% | 3.438% | 2.532% |
| Mean ES | 3.101% | 5.964% | 3.259% |
| Breach count | 134 | 84 | 96 |
| Breach rate | 2.420% | 1.517% | 1.733% |
| Target breach rate | 1.000% | 1.000% | 1.000% |

A correctly calibrated 99% VaR model should have a breach rate of approximately 
1% over repeated observations, subject to sampling variation.

The Gaussian model recorded the highest breach rate, at 2.420%. The Student-t 
model reduced the breach rate to 1.517%, a reduction of approximately 37.3% in 
the number of observed breaches. This supports the hypothesis that allowing for 
heavy-tailed returns can improve the representation of extreme losses.

The GARCH-Student-t model recorded a breach rate of 1.733%. Although 
this improves on the Gaussian benchmark, it does not outperform the static 
Student-t model on this metric.

The mean VaR and ES figures also reveal an important distinction. The Student-t 
model produced the largest average risk forecasts. A larger forecast is not necessarily better: the objective is to forecast risk accurately, not simply to maximize the predicted loss threshold.

## 3. Statistical Backtesting

### 3.1 Kupiec unconditional coverage test

The Kupiec test evaluates whether the observed breach frequency is consistent 
with the nominal 1% VaR breach probability.

All three models reject the null hypothesis of correct unconditional coverage 
at the 5% significance level. The Gaussian model has the largest coverage test 
statistic, while Student-t has the smallest.

This indicates that the observed breach frequencies differ materially from the 
nominal target, particularly for the Gaussian benchmark.

### 3.2 Christoffersen independence test

The independence test examines whether VaR breaches are independent over time.

| Model | Independence LR | p-value |
|---|---:|---:|
| Gaussian | 31.853 | <0.001 |
| Student-t | 13.215 | 0.000278 |
| GARCH-Student-t | 2.461 | 0.117 |

The Gaussian and Student-t models reject the independence null hypothesis 
at the 5% level, providing evidence that their breaches cluster over time.

The GARCH-Student-t model does not reject independence at the 5% level. 
This is consistent with the idea that time-varying volatility can help account for changing market risk. However, failure to reject independence is not proof that breaches are independent.

### 3.3 Conditional-coverage test

The conditional-coverage test combines the unconditional 
coverage and independence statistics:

\[
LR_{\mathrm{CC}}=LR_{\mathrm{UC}}+LR_{\mathrm{IND}}
\]

Under the null hypothesis, the statistic follows a chi-squared 
distribution with two degrees of freedom.

| Model | Conditional-coverage LR | p-value |
|---|---:|---:|
| Gaussian | 112.556 | 3.62 × 10⁻²⁵ |
| Student-t | 26.113 | 2.14 × 10⁻⁶ |
| GARCH-Student-t | 27.148 | 1.27 × 10⁻⁶ |

All three models reject conditional coverage at the 5% level.

The Student-t model has the lowest conditional-coverage statistic, 
but it still fails the joint test. GARCH-Student-t does not achieve 
correct full-sample conditional coverage despite showing less evidence 
of breach clustering.

These tests evaluate each model against its calibration assumptions; 
they do not constitute a direct paired test proving that one model is statistically 
superior to another.

## 4. Rolling Backtest Performance
### Rolling VaR Breach Rates

![Rolling 250-day VaR breach rates](figures/rolling_breach_rates.png)

Figure 1 shows the rolling 250-observation VaR breach rate for each model, compared with 
the theoretical 1% target for a correctly calibrated 99% VaR forecast. 
The rolling window illustrates how model performance changes over time and whether 
breaches become concentrated during periods of market stress.

The clustering of breaches during volatile periods suggests that static 
distributional models struggle to adapt to changing market conditions. 
Although the GARCH-Student-t model incorporates time-varying volatility, 
its performance should be evaluated using both breach frequency and formal 
backtesting results.


## 5. Crisis-Period Analysis

### Introduction and Crisis-results table
To assess how the models perform under extreme market conditions, the 
analysis focuses on two periods of severe market stress: the 2008 global 
financial crisis and the 2020 COVID-19 market crash. These periods provide 
a useful comparison of model performance during episodes of heightened 
volatility and substantial downside risk.

The table below reports the observed 99% VaR breach rates for each model 
during these periods. Under correct calibration, approximately 1% of 
observations would be expected to exceed the VaR threshold. Comparing 
the observed rates therefore provides an indication of how effectively 
each model captures downside risk during market stress.

| Period | Observations | Gaussian | Student-t | GARCH-Student-t |
|---|---:|---:|---:|---:|
| September 2008–June 2009 | 209 | 13.40% | 10.05% | 1.91% |
| February–May 2020 | 72 | 22.22% | 18.06% | 2.78% |

### 2008 Financial Crisis

The Gaussian model recorded a 13.40% breach rate, while Student-t reduced this 
to 10.05%. Both substantially exceeded the 1% target.

GARCH-Student-t recorded a much lower breach rate of 1.91%. Its mean VaR during 
this period was 7.26%, reflecting substantially more conservative risk forecasts.

### 2020 COVID-19 crash

The COVID-19 crash provides an especially clear illustration of differences in 
model behavior. The Gaussian model recorded a 22.22% breach rate, compared with 
18.06% for Student-t.

GARCH-Student-t recorded a breach rate of 2.78%. Its mean VaR was 8.07%, and its 
maximum VaR forecast reached 23.69%.

The GARCH model responded strongly to the extreme volatility observed during 
March 2020. This substantially reduced breaches, but it also generated very 
large risk forecasts.

### Comparison Across Market Crises

![VaR breach rates during the 2008 and 2020 crises](figures/crisis_breach_comparison.png)

Figure 3 compares the observed VaR breach rates during the 2008 financial crisis 
and the 2020 COVID-19 crash. The GARCH-Student-t model records substantially fewer 
breaches in both periods than the Gaussian and static Student-t models.

This suggests that incorporating time-varying volatility can improve risk protection 
during periods of market stress. Nevertheless, the results cover two specific crisis 
windows and do not establish that GARCH is the best model under all market conditions. 
Its performance must also be considered alongside its forecast magnitude and full-sample 
backtesting results.

### VaR Forecasts During the COVID-19 Crash

![S&P 500 VaR forecasts during the COVID-19 crash](figures/var_covid_crash.png)

Figure 2 compares realized daily losses with the three models' 
99% VaR forecasts during the COVID-19 market crash. The Gaussian 
and Student-t models use rolling historical estimation windows 
but do not explicitly model conditional volatility. In contrast, 
the GARCH-Student-t model adjusts its risk forecast in response 
to recent market movements.

The GARCH model's sharp increase in VaR during the crash illustrates 
its responsiveness to rapidly changing volatility. However, higher 
VaR forecasts also indicate a more conservative assessment of risk, 
so lower breach rates should not automatically be interpreted as 
superior overall performance.

### Interpretation

The crisis results suggest that dynamic volatility modeling can be valuable 
during periods of rapidly changing market conditions. However, the selected 
crisis windows are relatively short, particularly the 72-observation COVID 
period. The results should therefore be treated as descriptive evidence 
rather than definitive proof of model superiority.

The low GARCH breach rates should also be interpreted alongside the size of 
its forecasts. A model can reduce breaches by predicting very large losses, 
but this may make it overly conservative outside the most extreme conditions.

## 6. Financial Interpretation

The results highlight three distinct aspects of risk forecasting.

**Distributional assumptions:** The Gaussian benchmark struggles to capture 
the frequency of extreme returns. The Student-t model's heavier tails improve 
the observed breach frequency.

**Time-varying volatility:** GARCH-Student-t responds to volatility shocks 
and generates much larger forecasts during crisis periods. Its independence-test 
result suggests that the timing of breaches differs from the two static 
distributional approaches.

**Calibration versus conservatism:** The Student-t model performs best on 
full-sample breach frequency, whereas GARCH-Student-t produces much lower 
breach rates during the selected crises. Neither result alone establishes 
universal superiority. A useful risk model must balance calibration, tail-loss 
estimation, stability, and responsiveness to changing conditions.

## 7. Limitations

Several limitations should be considered when interpreting the findings.

1. **Single asset:** The analysis uses the S&P 500 and may not generalize to other equity markets, individual securities, or other asset classes.
2. **Historical estimation:** Rolling estimates depend on the selected 1,000-observation window and five-day refitting schedule.
3. **Limited crisis samples:** Crisis-period comparisons contain fewer observations than the full backtest and are sensitive to the selected date ranges.
4. **VaR and ES evaluation:** Breach frequency is not a complete measure of forecast quality. ES forecasts also require appropriate evaluation.
5. **Statistical uncertainty:** The formal tests assess calibration hypotheses but do not directly establish pairwise superiority between models.
6. **Extreme GARCH forecasts:** The very large forecasts during March 2020 may be useful responses to market stress, but they also raise questions about forecast conservatism and stability.

## 8. Conclusion

The results support the view that heavy-tailed return distributions can improve extreme downside risk 
forecasting relative to a Gaussian benchmark. The Student-t model reduced the full-sample VaR breach rate 
by approximately 37.3%.

GARCH-Student-t produced substantially lower breach rates during the 2008 and 2020 crisis windows, 
reflecting its ability to respond to changing volatility. However, its full-sample breach rate remained 
above the 1% target, and the model failed the conditional-coverage test.

Overall, no model achieved satisfactory full-sample conditional coverage. The findings demonstrate that 
accurate extreme-risk forecasting requires more than selecting a heavy-tailed distribution or adding a volatility 
model: calibration, temporal dependence, and the conservatism of risk forecasts must be evaluated together.
