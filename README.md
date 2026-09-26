# Federated and Explainable AI for Stroke-Status Classification

MSc Computing Research Project by **Harikrishna Bolla**  
Student ID: **C5054279**  
Sheffield Hallam University

## Project Overview

This project develops and evaluates a reproducible machine-learning workflow for **stroke-status classification** using a large public secondary dataset derived from the **CDC 2020 Behavioral Risk Factor Surveillance System (BRFSS)**.

The project compares Logistic Regression, Random Forest, XGBoost, a centralised neural network, and a simulated federated neural network using **FedAvg**. Evaluation goes beyond headline accuracy and includes stroke-class recall, precision, F1-score, macro-F1, balanced accuracy, ROC-AUC, PR-AUC, Brier score, calibration, subgroup performance across sex/age/race, and SHAP-based explainability.

The project is an **academic research prototype** only. It is not a clinical diagnostic system, medical device, prospective individual stroke predictor, or treatment recommendation tool.

## Research Question

**To what extent can machine-learning workflows provide reliable stroke-status classification from highly imbalanced public BRFSS-derived data when evaluated across discrimination, minority-class detection, probability calibration, subgroup performance and explainability, and what trade-offs arise between centralised and simulated federated implementation?**

## Aim

To design and critically evaluate a trustworthy and reproducible academic workflow for stroke-status classification using centralised and simulated federated machine learning.

## Objectives

1. Verify dataset provenance, licence, ethics and data quality.
2. Implement leakage-safe centralised machine-learning and neural-network baselines.
3. Simulate non-IID federated learning using FedAvg.
4. Evaluate models using imbalance-aware discrimination and calibration metrics.
5. Audit model behaviour across sex, age and race subgroups.
6. Explain the behaviour of the selected XGBoost model using SHAP.
7. Export reproducibility artefacts, model settings, metrics and visual evidence.

## Dataset

**Indicators of Heart Disease (2022 UPDATE)**  
Kaggle publisher: **Kamil Pytlak**  
Selected file: `heart_2020_cleaned.csv`

Dataset link:  
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data

### Dataset characteristics

- Records: **319,795**
- Variables: **18**
- Target: `Stroke`
- Stroke-positive prevalence: **3.774%**
- Missing cells: **0**
- Licence: **CC0: Public Domain**
- Source: cleaned derivative of the **CDC 2020 BRFSS**

### Dataset SHA-256

```text
90a1e2e8bdcb4aba54314a41d35d2ac3be4763e459df33951e08e415f527f5a8
```

## Repository Structure

```text
.
├── FINAL_UREC1_Federated_Explainable_Stroke_Kaggle.ipynb
├── README.md
├── best_central_model.joblib
├── central_locked_test_metrics.csv
├── central_nn_history.csv
├── central_nn_state.pt
├── central_validation_metrics.csv
├── central_vs_federated_nn.csv
├── confusion_matrix_Centralised_NN.csv
├── confusion_matrix_Federated_NN_nonIID_FedAvg.csv
├── data_dictionary.csv
├── dataset_audit.csv
├── dataset_provenance.json
├── dataset_quality_summary.csv
├── distribution_AgeCategory.csv
├── distribution_Race.csv
├── distribution_Sex.csv
├── distribution_Stroke.csv
├── experiment_settings.json
├── federated_client_distribution.csv
├── federated_history.csv
├── federated_nn_state.pt
├── figure_manifest.csv
├── FINAL_model_comparison.csv
├── hardware_environment.json
├── nn_preprocessor.joblib
├── numeric_summary.csv
├── output_manifest.csv
├── plausibility_checks.csv
├── RUN_SUMMARY.txt
├── shap_global_importance.csv
├── software_versions.csv
├── software_versions.json
├── split_indices.npz
├── split_summary.csv
├── stroke_percentages_by_AgeCategory.csv
├── stroke_percentages_by_Race.csv
├── stroke_percentages_by_Sex.csv
├── subgroup_fairness_metrics.csv
├── target_distribution.csv
└── UREC1_compliance_checklist.csv
```

## Main Notebook

The main executable notebook is:

```text
FINAL_UREC1_Federated_Explainable_Stroke_Kaggle.ipynb
```

It contains the full workflow from provenance checks and EDA through centralised modelling, federated learning, calibration, subgroup analysis, SHAP and artefact export.

## End-to-End Workflow

1. Load and verify the exact Kaggle dataset file.
2. Record provenance, licence and SHA-256.
3. Create the 18-variable data dictionary.
4. Audit missingness, duplicates, plausibility and class imbalance.
5. Perform exploratory analysis.
6. Create a stratified **80/10/10** train/validation/test split.
7. Fit preprocessing on training data only.
8. Train Logistic Regression, Random Forest and XGBoost models.
9. Select decision thresholds on validation data only.
10. Evaluate centralised models on the locked test set.
11. Train a centralised neural network.
12. Create five artificial non-IID federated clients.
13. Train a federated neural network using FedAvg.
14. Compare centralised and federated neural-network performance.
15. Evaluate calibration using Brier score and calibration curves.
16. Audit sex, age and race subgroup performance.
17. Generate SHAP explanations for XGBoost.
18. Save all metrics, settings, models and reproducibility artefacts.

## Preprocessing

### Numeric variables

- `BMI`
- `PhysicalHealth`
- `MentalHealth`
- `SleepTime`

These are standardised using `StandardScaler`.

### Categorical variables

Categorical features are transformed with `OneHotEncoder`.

### Leakage prevention

```text
Raw data
   ↓
Train / validation / test split
   ↓
Fit preprocessing on training data only
   ↓
Transform validation and test sets
```

The locked test set is not used for model or threshold selection.

## Data Split

- Training: **80%**
- Validation: **10%**
- Locked test: **10%**
- Locked-test records: **31,980**

Files:
- `split_summary.csv`
- `split_indices.npz`

## Models Implemented

### Logistic Regression
Interpretable linear baseline.

### Random Forest
Non-linear ensemble baseline based on multiple decision trees.

### XGBoost
Gradient-boosted decision trees used because they perform strongly on structured tabular data and work well with TreeSHAP.

Configurations:
- `XGBoost_standard`
- `XGBoost_class_aware`

### Centralised Neural Network

```text
50 encoded inputs
→ 64 hidden units
→ 32 hidden units
→ 1 binary output
```

Uses ReLU, dropout, AdamW and weighted binary cross-entropy.

### Federated Neural Network

- Artificial clients: **5**
- Dirichlet alpha: **0.5**
- Communication rounds: **6**
- Local training: **1 epoch/client/round**
- Aggregation: **FedAvg**

This is a simulated statistical heterogeneity experiment, not a real hospital federation.

## Main Libraries

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- PyTorch
- SHAP
- Matplotlib
- Joblib

Environment evidence:
- `software_versions.csv`
- `software_versions.json`
- `hardware_environment.json`

## Evaluation Metrics

Because the positive class is rare, the project does not rely on accuracy alone.

Metrics include:

- Accuracy
- Precision
- Recall / Sensitivity
- F1-score
- Macro-F1
- Balanced accuracy
- ROC-AUC
- PR-AUC
- Brier score
- Calibration curves
- Confusion matrices

## Final Locked-Test Results

| Model | Accuracy | Stroke Recall | Positive F1 | Balanced Accuracy | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Centralised NN | 0.9224 | 0.3389 | **0.2479** | **0.6421** | **0.8269** | **0.1771** | 0.1730 |
| XGBoost standard | 0.9201 | **0.3397** | 0.2430 | 0.6413 | 0.8245 | 0.1695 | **0.0336** |
| XGBoost class-aware | 0.9191 | 0.3347 | 0.2379 | 0.6384 | 0.8207 | 0.1679 | 0.1607 |
| Federated NN | **0.9248** | 0.3190 | 0.2426 | 0.6338 | 0.8214 | 0.1675 | 0.0628 |
| Logistic Regression balanced | 0.9195 | 0.3355 | 0.2394 | 0.6390 | 0.8227 | 0.1638 | 0.1720 |
| Random Forest balanced | 0.9234 | 0.2212 | 0.1789 | 0.5861 | 0.7876 | 0.1195 | 0.0853 |

Complete results:
- `FINAL_model_comparison.csv`

## Main Findings

- **XGBoost standard** was the strongest conventional centralised ML model and had the best probability calibration.
- **Centralised NN** achieved the strongest ROC-AUC, PR-AUC, positive-class F1 and balanced accuracy.
- **Federated NN** retained similar headline accuracy but lower stroke recall and PR-AUC under severe non-IID heterogeneity.
- **Random Forest** was weaker on minority-class detection.
- The results show why trustworthy model assessment must consider discrimination, minority recall and calibration together.

## Confusion Matrices

### XGBoost Standard

```text
TN = 29,016
FP = 1,757
FN = 797
TP = 410
```

### Centralised Neural Network

```text
TN = 29,089
FP = 1,684
FN = 798
TP = 409
```

### Federated Neural Network

```text
TN = 29,191
FP = 1,582
FN = 822
TP = 385
```

Files:
- `confusion_matrix_Centralised_NN.csv`
- `confusion_matrix_Federated_NN_nonIID_FedAvg.csv`

## Calibration

Calibration is evaluated with the Brier score and calibration curves.

Important finding:

> Good discrimination does not automatically mean well-calibrated probabilities.

Standard XGBoost achieved the strongest calibration:

```text
Brier score = 0.0336
```

## Federated Learning

Relevant artefacts:
- `federated_client_distribution.csv`
- `federated_history.csv`
- `federated_nn_state.pt`

The client split is intentionally severe and should be interpreted as a **non-IID stress test**.

## Subgroup Analysis

Subgroups:
- Sex
- AgeCategory
- Race

Main file:
- `subgroup_fairness_metrics.csv`

Supporting files:
- `distribution_Sex.csv`
- `distribution_AgeCategory.csv`
- `distribution_Race.csv`
- `stroke_percentages_by_Sex.csv`
- `stroke_percentages_by_AgeCategory.csv`
- `stroke_percentages_by_Race.csv`

Subgroup results are descriptive and do not constitute proof of algorithmic fairness.

## Explainability with SHAP

TreeSHAP is applied to `XGBoost_standard`.

Main file:
- `shap_global_importance.csv`

Important transformed features include heart-disease status, walking difficulty, general health, age, physical health, sleep time, smoking, BMI, mental health and diabetes.

SHAP explains model behaviour; it does **not** establish causal or clinical relationships.

## Data Quality Artefacts

- `data_dictionary.csv`
- `dataset_audit.csv`
- `dataset_quality_summary.csv`
- `plausibility_checks.csv`
- `numeric_summary.csv`
- `target_distribution.csv`
- `distribution_Stroke.csv`
- `distribution_Sex.csv`
- `distribution_AgeCategory.csv`
- `distribution_Race.csv`

## Reproducibility Artefacts

### `dataset_provenance.json`
Dataset title, publisher, selected file, licence, target, dimensions and SHA-256.

### `experiment_settings.json`
Seed, thresholds, split proportions, federated settings, learning rate, input dimension and environment.

### `split_indices.npz`
Exact train/validation/test indices.

### `split_summary.csv`
Split sizes and target prevalence.

### `software_versions.csv` / `software_versions.json`
Software versions.

### `hardware_environment.json`
Hardware/device information.

## Saved Model Artefacts

- `best_central_model.joblib`
- `nn_preprocessor.joblib`
- `central_nn_state.pt`
- `federated_nn_state.pt`

## Training Histories

- `central_nn_history.csv`
- `federated_history.csv`

## Manifests and Final Run Evidence

- `figure_manifest.csv`
- `output_manifest.csv`
- `RUN_SUMMARY.txt`
- `UREC1_compliance_checklist.csv`

## How to Run on Kaggle

### 1. Clone or download the repository

```bash
git clone https://github.com/c5054279/federated-explainable-ai-stroke-classification.git
```

### 2. Open the notebook

```text
FINAL_UREC1_Federated_Explainable_Stroke_Kaggle.ipynb
```

### 3. Attach the Kaggle dataset

https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data

### 4. Confirm the file

```text
heart_2020_cleaned.csv
```

### 5. Run all cells

The notebook generates the complete analysis, models, metrics and reproducibility artefacts.

## Reproducibility

Supported by:

- fixed seed,
- dataset SHA-256,
- exact split indices,
- software-version records,
- hardware record,
- saved model states,
- saved preprocessing object,
- experiment settings,
- metric tables,
- output and figure manifests.

Primary seed:

```text
42
```

## Ethics and Research Boundaries

This project uses secondary public-domain data only.

No participants were recruited, contacted, interviewed, surveyed or re-identified.

The system must not be interpreted as:
- a clinical diagnostic system,
- a medical device,
- a prospective individual stroke predictor,
- a treatment recommendation system,
- causal medical evidence.

It classifies **reported stroke status** in cross-sectional BRFSS-derived records.

## Limitations

1. Self-reported data.
2. Cross-sectional design.
3. Severe class imbalance.
4. No full BRFSS complex-survey weighting variables in the cleaned derivative.
5. Simulated rather than real multi-institutional federation.
6. Artificial clients do not represent hospitals.
7. Some subgroup estimates are unstable.
8. SHAP is not causal explanation.
9. No external clinical validation.
10. No clinical deployment.

## Future Work

- external validation,
- probability recalibration,
- confidence intervals,
- repeated validation,
- stronger federated algorithms,
- sensitivity analysis for client count and Dirichlet alpha,
- real decentralised clients,
- differential privacy,
- secure aggregation,
- additional fairness metrics,
- approved expert evaluation.

## GitHub Repository

https://github.com/c5054279/federated-explainable-ai-stroke-classification

## Artefact Traceability

| Evidence | Repository artefact |
|---|---|
| Dataset provenance | `dataset_provenance.json` |
| Data dictionary | `data_dictionary.csv` |
| Data-quality audit | `dataset_audit.csv` |
| Plausibility checks | `plausibility_checks.csv` |
| Split evidence | `split_summary.csv`, `split_indices.npz` |
| Central model results | `central_locked_test_metrics.csv` |
| Central NN history | `central_nn_history.csv` |
| Federated client design | `federated_client_distribution.csv` |
| FedAvg history | `federated_history.csv` |
| Central vs federated comparison | `central_vs_federated_nn.csv` |
| Final model comparison | `FINAL_model_comparison.csv` |
| Subgroup evaluation | `subgroup_fairness_metrics.csv` |
| SHAP evidence | `shap_global_importance.csv` |
| Software environment | `software_versions.json` |
| Experiment configuration | `experiment_settings.json` |
| Final run summary | `RUN_SUMMARY.txt` |
| UREC1 checks | `UREC1_compliance_checklist.csv` |

## Author

**Harikrishna Bolla**  
MSc Computing  
Sheffield Hallam University  
Student ID: **C5054279**

## Licence and Use

The source dataset is recorded in the project documentation as **CC0: Public Domain**.

The repository artefacts are provided for academic research and reproducibility. Third-party packages retain their own licences.

## Disclaimer

This repository is for **academic research and educational purposes only**.

It must not be used for medical diagnosis, treatment recommendations, individual future-stroke prediction or healthcare deployment without independent validation and appropriate clinical, ethical and regulatory review.
