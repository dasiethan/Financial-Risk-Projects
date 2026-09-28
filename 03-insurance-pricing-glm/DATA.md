# Data

| | |
|---|---|
| Datasets | `freMTPL2freq` (policies and claim counts) and `freMTPL2sev` (claim amounts) |
| Provider | OpenML, dataset ids **41214** and **41215** |
| Pages | https://www.openml.org/d/41214 and https://www.openml.org/d/41215 |
| Licence (OpenML) | CC0 (public domain dedication) |
| Original source | R package `CASdatasets` (A. Charpentier, C. Dutang), *Computational Actuarial Science with R* |
| Size | 678,013 policies × 12 columns; 26,639 claim records; about 7 MB cached |

## Download

`python main.py` calls `sklearn.datasets.fetch_openml`, which downloads both tables on the
first run and caches them under `data/`. The `data/` folder is not versioned.

## Variables

- `IDpol`: policy id, the key linking the two tables
- `ClaimNb`: number of claims declared during the exposure period
- `Exposure`: exposure in years
- `Area`: area code (A to F, from rural to urban)
- `VehPower`: vehicle power (ordered categories)
- `VehAge`: vehicle age in years
- `DrivAge`: driver age in years
- `BonusMalus`: French bonus-malus level (50 = maximum bonus, above 100 = malus)
- `VehBrand`: vehicle brand (anonymised)
- `VehGas`: fuel (Diesel or Regular)
- `Density`: inhabitants per km² of the driver's city
- `Region`: French region, pre-2016 classification
- `ClaimAmount` (severity table): amount of each claim payment, in €

## Known data issues (handled in `src/data.py`)

- 9,116 policies with `ClaimNb > 0` have no record in the severity table.
- 195 severity records refer to policies absent from the frequency table. They are dropped.
- Some exposures are above 1 year and some claim counts are implausibly high (up to 16).
