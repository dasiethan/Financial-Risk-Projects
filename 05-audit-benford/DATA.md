# Data

| | |
|---|---|
| Dataset | *Données Essentielles de la Commande Publique (DECP) – Arrêté du 22/12/2022 – Marchés* |
| Publisher | Ministères économiques et financiers (France) |
| Pages | https://data.economie.gouv.fr/explore/dataset/decp-2022-marches-valides/ and the data.gouv.fr page of the same name |
| Licence | Licence Ouverte / Open Licence v2.0 (Etalab), which allows free reuse with attribution |
| Extract used | Contracts with `year(datenotification) = 2025` and `datepublicationdonnees <= 2026-09-01` |
| Size | 253,863 rows, about 60 MB (CSV) |

## Download

`python main.py` calls the Opendatasoft export API once, with the filters above and the
selected columns only, and saves `data/decp_2025.csv`. The `data/` folder is not versioned.
The equivalent request is:

```
https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/decp-2022-marches-valides/exports/csv
  ?select=id,acheteur_id,nature,procedure,codecpv,montant,dureemois,datenotification,
          datepublicationdonnees,offresrecues,formeprix,titulaire_id_1,objet
  &where=year(datenotification)=2025 and datepublicationdonnees<=date'2026-09-01'
  &delimiter=;
```

## Fields used

- `acheteur_id`, `titulaire_id_1`: SIRET numbers of the buyer and the (first) supplier
- `id`: the buyer's contract identifier
- `procedure`: award procedure
- `codecpv`: Common Procurement Vocabulary code; its first 2 digits give the division
- `montant`: initial amount excluding VAT, in €. For framework agreements, often the maximum
- `dureemois`: duration in months
- `datenotification`, `datepublicationdonnees`: notification and publication dates
- `offresrecues`: number of offers received (free text in some records, e.g. "MQ NC")
- `formeprix`: price type (unit, lump sum, mixed)
- `objet`: description of the purchase, used only to read the top anomalies

## Caveats

The DECP are declared by thousands of buyers through different platforms. Records are
corrected and added over time, so a later download with the same filters may differ
slightly.

Identifiers of legal entities are public information. This project reports only aggregated
results and does not single out any buyer or supplier.
