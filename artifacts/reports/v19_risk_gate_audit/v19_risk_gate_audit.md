# v19 Risk Gate Learnability Audit

## Default Expert Audit

Selected default expert: `itransformer`

### train

| model | mean_mse | mean_regret | p90_regret | p95_regret |
| --- | ---: | ---: | ---: | ---: |
| patchtst | 11.864667 | 2.981496 | 8.180658 | 11.610204 |
| itransformer | 10.683552 | 1.800382 | 5.300784 | 7.817403 |
| arima | 89036571.938225 | 89036563.055055 | 19.374848 | 30.976703 |
| chronos2 | 13.415023 | 4.531853 | 11.881171 | 18.156549 |

### val

| model | mean_mse | mean_regret | p90_regret | p95_regret |
| --- | ---: | ---: | ---: | ---: |
| patchtst | 5.069778 | 1.401408 | 3.719355 | 6.707573 |
| itransformer | 5.003042 | 1.334672 | 3.086696 | 8.335725 |
| arima | 5.849601 | 2.181231 | 5.524867 | 8.747692 |
| chronos2 | 5.270413 | 1.602044 | 4.354523 | 6.682753 |

### test

| model | mean_mse | mean_regret | p90_regret | p95_regret |
| --- | ---: | ---: | ---: | ---: |
| patchtst | 8.084212 | 2.369447 | 6.726862 | 9.227681 |
| itransformer | 7.805982 | 2.091217 | 6.380462 | 9.184467 |
| arima | 9.883089 | 4.168324 | 12.502456 | 18.242445 |
| chronos2 | 8.881174 | 3.166409 | 9.341510 | 12.583296 |

## Risk Gate Audit

Visible inputs use the current Turn-2 feature view only: the routing feature snapshot derived from historical data. Hidden oracle quantities such as `best_model`, `default_error`, and `route_margin_rel` are not given to the classifier.

The environment does not include `xgboost` or `lightgbm`, so `HistGradientBoostingClassifier` is used as the tree-boosting fallback for the third audit model.

### tau=0.15

Label distribution: train={'0': 68, '1': 132}, val={'0': 26, '1': 38}, test={'0': 55, '1': 73}

| model | split | auc | f1 | balanced_acc | predicted_distribution |
| --- | --- | ---: | ---: | ---: | --- |
| logistic_regression | val | 0.4990 | 0.6190 | 0.4575 | {'0': 18, '1': 46} |
| logistic_regression | test | 0.4640 | 0.5493 | 0.4944 | {'0': 59, '1': 69} |
| random_forest | val | 0.4696 | 0.7273 | 0.4929 | {'0': 3, '1': 61} |
| random_forest | test | 0.4367 | 0.6474 | 0.4836 | {'0': 28, '1': 100} |
| hist_gradient_boosting | val | 0.4484 | 0.6452 | 0.4140 | {'0': 9, '1': 55} |
| hist_gradient_boosting | test | 0.4677 | 0.6341 | 0.5016 | {'0': 37, '1': 91} |

Gate pass at tau=0.15: `False`

### tau=0.20

Label distribution: train={'0': 84, '1': 116}, val={'0': 32, '1': 32}, test={'0': 64, '1': 64}

| model | split | auc | f1 | balanced_acc | predicted_distribution |
| --- | --- | ---: | ---: | ---: | --- |
| logistic_regression | val | 0.4717 | 0.4865 | 0.4062 | {'0': 22, '1': 42} |
| logistic_regression | test | 0.4890 | 0.4651 | 0.4609 | {'0': 63, '1': 65} |
| random_forest | val | 0.5386 | 0.6353 | 0.5156 | {'0': 11, '1': 53} |
| random_forest | test | 0.5404 | 0.5594 | 0.5078 | {'0': 49, '1': 79} |
| hist_gradient_boosting | val | 0.4414 | 0.5476 | 0.4062 | {'0': 12, '1': 52} |
| hist_gradient_boosting | test | 0.5508 | 0.5385 | 0.5312 | {'0': 62, '1': 66} |

Gate pass at tau=0.20: `False`

## Gate 0 Decision

Gate 0 failed. Learned Risk Gate is not learnable enough from the current Turn-2 visible state.

Per v19, Turn 2 should switch to fixed expand (`default_risky`) and the main decision should move to Turn 3 final selection.

