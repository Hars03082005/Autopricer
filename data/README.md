# data/

This folder contains the **full combined dataset** for reference and analysis only.

## data.csv

| Split      | Rows         |
|------------|--------------|
| train      | 17,632 |
| valid      | 3,778 |
| test       | 3,748  |
| **Total**  | **25,158** |

> **WARNING**: This file contains ALL splits including the held-out test set.
> It **must not** be used for model training, hyperparameter tuning, or any
> other form of model fitting. Use only for exploratory analysis, reporting,
> or the production comparable-search engine.

A `split` column is included on every row to indicate its original split origin.

SHA-256: `4f228903c464edba097a7afb04c593c1c76caa05f1096b938a1a25c17e0f9de5`
