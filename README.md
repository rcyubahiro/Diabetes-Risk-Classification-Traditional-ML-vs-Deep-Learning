# Diabetes Risk Classification: Traditional ML vs Deep Learning: **Predicting Diabetes Risk from Population Health Survey Indicators**

ALU Machine Learning summative (Model Training and Evaluation).

Predicts whether a survey respondent has prediabetes or diabetes from 21 low-cost health and lifestyle indicators, comparing scikit-learn models against TensorFlow/Keras networks built on `tf.data` pipelines.

## Links

- Report: https://docs.google.com/document/d/1Jon5gHEgyYJii7s2uyIeQGiAf2G0RMVy8rAIzW6BOJs/edit?usp=sharing 
- Google Drive folder (raw dataset zip, best models): (https://drive.google.com/drive/folders/1Z3mc9aF6JI6_NOKZ3VIZW2C_B-6Alt2w?usp=sharing)
- Video presentation: https://youtu.be/k4E65LQEBsA

## Problem and data

- **Task:** binary classification, `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes).
- **Dataset:** CDC Diabetes Health Indicators, UCI Machine Learning Repository (ID 891), from the 2015 BRFSS survey. About 253,680 rows, 21 features. Open licence (CC BY 4.0).
- **Why:** screening resources are limited in underserved communities, and a model that works from cheap survey answers could help prioritise who gets tested.
- The raw data is **not** in this repository. The notebook downloads it automatically with `ucimlrepo`; a zipped copy is in the Google Drive folder.

## Repository layout

```
diabetes_classification.ipynb   main notebook (run top to bottom)
requirements.txt                dependencies
artifacts/
  experiment_results.csv        one row per experiment (hyperparameters, splits, metrics, observations)
  test_results.csv              final test-set metrics for the best model of each family
  subgroup_errors.csv           error rates by age, income and sex
figures/                        figures used in the report
logs/                           W&B offline runs and TensorBoard logs backing the experiment table
```

Trained models and the dataset zip live in Google Drive, not in this repo.

## How to run

**Google Colab (recommended):** upload or open `diabetes_classification.ipynb`, then Runtime > Run all. A GPU is not required. The first cell installs the extra packages.

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook diabetes_classification.ipynb
```

Set `FAST_DEBUG = True` in the setup cell for a 2-minute smoke test on a small subsample; set it back to `False` for the full run that produces the reported results.

## Experiments

| ID | Approach | What changes |
|---|---|---|
| E1 | Logistic regression | linear baseline |
| E2 | Random forest, default | deep, high-variance reference |
| E3 | Random forest, tuned | randomised search, depth and leaf constraints, class weights |
| E4 | HistGradientBoosting, tuned | boosting with regularisation and early stopping |
| E5 | SVM (RBF), subsample | non-linear kernel, compute trade-off |
| E6 | Keras Sequential 64-32 | plain neural baseline |
| E7 | Keras Sequential 256-128-64 | batch norm, dropout, L2, early stopping |
| E8 | Keras Functional, residual block | lower learning rate, LR scheduling |
| E9 | E8 + class weights | recall versus precision trade-off |
| E10 | E4 + engineered features | feature-engineering ablation (trees) |
| E11 | E8 + engineered features | feature-engineering ablation (network) |

Model selection and the decision threshold use the validation set only; the test set is evaluated once at the end.

## Experiment tracking

Every run is logged to Weights & Biases in **offline mode** and, for the networks, to TensorBoard.

```bash
tensorboard --logdir logs/tensorboard
wandb sync logs/wandb/offline-run-*      # optional, uploads offline runs
```

## Reproducibility

Seeds are fixed (`SEED = 42`) for Python, NumPy and TensorFlow; the split is stratified and deterministic; preprocessing statistics are fitted on the training set only.

## Data source

Centers for Disease Control and Prevention, Behavioral Risk Factor Surveillance System 2015; distributed via the UCI Machine Learning Repository (CDC Diabetes Health Indicators).
