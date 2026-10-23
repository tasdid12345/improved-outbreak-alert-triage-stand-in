# Class-balanced calibrated boosting vs logistic regression for rare-positive alerts (Adult proxy)

Social-media disease surveillance systems must flag rare but important signals among large volumes of noisy posts, as surveyed in "A survey of social web mining applications for disease outbreak detection" (2015). Such alert classifiers operate under heavy class imbalance, so the choice of model and its calibration matters for how reliably rare positives are caught. This project asks whether a class-balanced, calibrated gradient-boosting classifier flags rare positives more reliably than a standard logistic-regression alert classifier.

**Question:** Can a class-balanced, calibrated gradient-boosting classifier with per-alert feature attributions flag rare positive signals more reliably than a standard logistic-regression alert classifier on a public imbalanced tabular dataset?

**Dataset:** UCI Adult income (downloaded via `ucimlrepo`), with income >50K as the positive class. This is a stand-in proxy and is not the paper's own dataset.

**Baseline:** Standard logistic regression with default settings on scaled numeric and one-hot categorical features.

**Improved:** LightGBM with balanced class weights, a small cross-validated hyperparameter search on the training split, and sigmoid calibration.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

## Results

Results are written to `results.json` (and `figures/f1_comparison.png`).

## Reference

This project is motivated by the following paper:

> G Bello-Orgaz, J Hernandez-Castro (2015). *A survey of social web mining applications for disease outbreak detection*. https://link.springer.com/chapter/10.1007/978-3-319-10422-5_36

Paper author: Prof. Julio Hernandez-Castro, University of Kent (J.C.Hernandez-Castro@kent.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
