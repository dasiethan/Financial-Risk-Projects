# Financial Risk Projects

Personal projects in **credit, market and insurance risk, audit analytics and ALM**, built
only on public data. They were developed alongside my Master 2 in Economic and Financial
Risk Engineering (Université de Bordeaux).

Every project is self-contained. Each has:

- its own README: objective, data, method, results, limitations, how to reproduce;
- a `DATA.md` describing the data source and licence;
- a pinned `requirements.txt`;
- a single `main.py` that runs end to end.

All figures and numbers in the READMEs are produced by that script. Weak or inconclusive
results are reported as such in each "Limitations" section.

| # | Project | What it does | Stack |
|---|---|---|---|
| 01 | [Credit risk scoring](01-credit-risk-scoring/) | Probability-of-default model on 30,000 credit card clients: WoE/IV scorecard (logistic regression, points scaling) vs LightGBM, with AUC/Gini/KS, calibration and PSI stability monitoring | pandas, statsmodels, scikit-learn, LightGBM, matplotlib |
| 02 | [Market risk: VaR & ES](02-market-risk-var-es/) | One-day VaR 99% and ES 97.5% of a French equity portfolio (historical, Normal, Student-t, Monte Carlo, GARCH-t, filtered historical simulation), backtested over 14 years with Kupiec, Christoffersen and the Basel traffic light | pandas, SciPy, arch, yfinance, matplotlib |
| 03 | [Motor insurance pricing](03-insurance-pricing-glm/) | Pure premium for 678,000 French motor liability policies: Poisson frequency × Gamma severity GLMs (relativities, factor selection, large-claim loading) vs LightGBM, compared on deviance, Lorenz/Gini and permutation importance | statsmodels, scikit-learn, LightGBM, matplotlib |
| 04 | [ESG vs conventional funds](04-esg-fama-french/) | CAPM, Fama-French 3 and 5 factors + momentum regressions (Newey-West) of four US ESG ETFs against conventional twins: alpha of the ESG-minus-conventional portfolio, factor tilts, statistical power | statsmodels, pandas, yfinance, Kenneth French Data Library |
| 05 | [Audit analytics on public procurement](05-audit-benford/) | Benford first/two-digit tests (Nigrini MAD), round-number and threshold-bunching tests, nine rule-based audit controls and an Isolation Forest on 252,000 French public contracts (DECP 2025) | pandas, SciPy, scikit-learn, data.economie.gouv.fr API |
| 06 | [Yield curve & ALM](06-alm-nelson-siegel/) | Nelson-Siegel fits of the ECB AAA euro curve every month since 2004 (level, slope, curvature factors), then an ALM application: durations, convexity, factor durations and surplus stress tests showing what duration matching does and does not hedge | NumPy, SciPy, pandas, ECB Data Portal API |

## Running a project

```bash
cd 01-credit-risk-scoring
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Datasets are downloaded on the first run and are not versioned (see each `DATA.md`). Each
project was run end to end from a fresh virtual environment with Python 3.14 on Windows.

## Licence

Code: MIT (see [LICENSE](LICENSE)). Datasets remain under their own licences.
