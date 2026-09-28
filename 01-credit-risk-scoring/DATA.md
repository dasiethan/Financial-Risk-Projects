# Data

| | |
|---|---|
| Dataset | Default of Credit Card Clients |
| Provider | UCI Machine Learning Repository (dataset id 350) |
| Page | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| Direct file | https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Citation | Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H |
| Size | 30,000 rows × 25 columns, about 5 MB (.xls) |

## Download

`python main.py` downloads the archive on first run and stores
`data/default_of_credit_card_clients.xls`. The `data/` folder is not versioned.

To download it manually, unzip the archive above and place the `.xls` file in `data/` under
that name.

## Variables (after renaming)

- `LIMIT_BAL`: credit limit (NT dollars)
- `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
- `PAY_1` … `PAY_6`: repayment status from September back to April 2005
  (-2 = no consumption, -1 = paid in full, 0 = revolving credit, k = k months of delay).
  `PAY_1` is named `PAY_0` in the original file.
- `BILL_AMT1` … `BILL_AMT6`: bill statement amounts (September back to April)
- `PAY_AMT1` … `PAY_AMT6`: amounts paid (September back to April)
- `default`: 1 if the client defaulted on the next payment (October 2005)
