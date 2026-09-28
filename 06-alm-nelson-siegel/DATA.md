# Data

| | |
|---|---|
| Source | European Central Bank, ECB Data Portal, dataset `YC` (euro area yield curves) |
| Page | https://data.ecb.europa.eu/data/datasets/YC |
| Methodology | https://www.ecb.europa.eu/stats/financial_markets_and_interest_rates/euro_area_yield_curves/ (technical notes) |
| Curve | Government bonds of all euro area issuers rated AAA, nominal, Svensson model, continuous compounding |
| Series | `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{3M,6M,9M,1Y…10Y,15Y,20Y,25Y,30Y}` (spot rates, %) and `…SV_C_YM.{BETA0,BETA1,BETA2,BETA3,TAU1,TAU2}` (Svensson parameters) |
| Period | 2004-09-06 (first available day) to 2025-12-31, business days |
| Size | 5,451 days × 23 series (about 2 MB) |

## Terms of use

ECB statistics may be reused, including for commercial purposes, provided the source is
acknowledged ("Source: ECB"). This project is non-commercial.

## Download

`python main.py` calls the ECB SDMX REST API once, with the series listed above, and caches
the result in `data/ecb_aaa_yield_curve.csv`. The `data/` folder is not versioned.

Example request:

```
https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y+SR_3M
  ?startPeriod=2004-09-06&endPeriod=2025-12-31&format=csvdata
```

## Known issue

On 2022-06-23, the published Svensson parameters and the published spot rates are not
consistent with each other (up to 5.6 bp). On every other day they match to machine
precision. The project uses the spot rates.
