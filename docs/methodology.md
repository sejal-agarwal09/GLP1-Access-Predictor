# GLP-1 Use Predictor: Methodology

## Study Objective

This project develops a machine-learning model to predict self-reported GLP-1 medication use among adults using publicly available data from the CDC/NCHS Round 7 Rapid Surveys.

The target variable is `GLP_MED12M`, which indicates whether a respondent reported taking a GLP-1 medication for diabetes or weight loss during the previous 12 months.

The model is intended as a research and educational prediction tool. It is not a clinical decision-support system and should not be used to determine whether an individual should receive a medication.

## Data Source

The analysis uses the CDC/NCHS Round 7 Rapid Surveys public-use dataset.

The Round 7 survey collected information during July-August 2025 and included questions about GLP-1 medication access and use.

The public-use dataset contains 8,118 records and 231 variables.

After excluding respondents with invalid or non-substantive target responses, 8,090 respondents remained for modeling.

## Outcome Definition

The primary outcome was:

`GLP_MED12M`

Coding:

- `0`: No GLP-1 use
- `1`: GLP-1 use

Non-substantive responses were excluded from the modeling dataset.

The unweighted prevalence of reported GLP-1 use was 12.06%.

Using the CDC final survey weight, the estimated population prevalence was 9.86% (95% CI: 9.04%-10.68%).

## Predictor Variables

The initial predictor set included:

- History of obesity
- History of diabetes
- Health insurance status
- Doctor visit during the previous 12 months
- Usual place for healthcare
- Age group
- Sex
- Race/ethnicity
- Census region
- Education
- Poverty status
- BMI category

Variables that directly described GLP-1 medication use or characteristics of the medication were excluded to reduce the risk of target leakage.

## Data Cleaning

CDC negative response codes representing skipped questions, refusals, unknown responses, or questions that were not asked were converted to missing values.

The target variable was restricted to valid responses of 0 or 1.

Missing predictor values were handled inside the machine-learning preprocessing pipeline using most-frequent imputation.

This preprocessing was fitted only on the training data through a scikit-learn pipeline to avoid information leakage.

Categorical predictors were one-hot encoded.

## Model Development

The dataset was divided into:

- 60% training
- 20% validation
- 20% final test

All splits were stratified by the GLP-1 outcome.

Four candidate models were evaluated:

1. Logistic regression
2. Class-weighted logistic regression
3. Random forest
4. Histogram-based gradient boosting

Models were compared using validation ROC-AUC, PR-AUC, and F1 score.

Because GLP-1 use represented a minority class, PR-AUC was emphasized alongside ROC-AUC.

The final model was selected using validation PR-AUC.

## Classification Threshold

The probability threshold was selected using the validation set rather than the final test set.

The selected threshold was 0.25, which produced the highest validation F1 score among the evaluated thresholds.

The final test set was then evaluated once using the selected model and threshold.

## Final Model Performance

The selected model was baseline logistic regression.

Final test performance:

- Accuracy: 0.865
- Precision: 0.452
- Recall: 0.554
- F1: 0.498
- ROC-AUC: 0.865
- PR-AUC: 0.479

The final test confusion matrix was:

- True negatives: 1,292
- False positives: 131
- False negatives: 87
- True positives: 108

Five-fold cross-validation on the training data produced a mean ROC-AUC of approximately 0.843 with a standard deviation of 0.014.

## Robustness Analysis

The logistic model was evaluated across five additional stratified random splits.

Across these splits:

- Mean ROC-AUC: 0.860
- ROC-AUC SD: 0.015
- Mean PR-AUC: 0.489
- PR-AUC SD: 0.028

These results indicate reasonably stable discrimination across different train/test partitions.

## Missing-Data Sensitivity Analysis

The primary analysis used most-frequent imputation for missing categorical predictors.

A complete-case sensitivity analysis was also performed by removing observations with missing predictor values.

Results:

| Method | ROC-AUC | PR-AUC |
|---|---:|---:|
| Most-frequent imputation | 0.868 | 0.477 |
| Complete-case analysis | 0.858 | 0.467 |

The difference was approximately 0.01 for both metrics, suggesting that model discrimination was reasonably robust to this missing-data handling choice.

The complete-case analysis contained fewer observations, so the two estimates should not be interpreted as perfectly equivalent comparisons.

## Survey-Weighted Analysis

Because the CDC Round 7 data come from a complex survey design, descriptive population estimates used the CDC-provided final combined survey weight (`WEIGHT`).

The analysis also incorporated:

- `P_STRATA_R` for survey strata
- `P_PSU_R` for primary sampling units

Variance estimation used the R `survey` package with Taylor-series linearization and CDC-recommended handling of strata containing a single PSU.

The weighted estimate of GLP-1 use was:

**9.86% (95% CI: 9.04%-10.68%)**

Survey weights were not directly incorporated into the scikit-learn predictive model. The weighted analysis was treated separately as a population-estimation component.

## Calibration

Calibration was evaluated on the final test set by comparing predicted probabilities with observed outcome frequencies across probability bins.

The model showed reasonably good overall calibration, although some lower-probability bins showed differences between predicted and observed rates.

The Brier score was approximately 0.081.

## Subgroup Analysis

Model performance was examined across demographic and clinical subgroups, including sex, age, BMI, and race/ethnicity.

Subgroup metrics should be interpreted cautiously when sample sizes or numbers of positive cases are small.

For example, the final test-set performance by sex showed similar ROC-AUC values for males and females, while recall was higher among females.

## Interpretation

The model identifies statistical patterns associated with reported GLP-1 medication use.

Important predictors included diabetes history, obesity history, BMI category, age, insurance status, and socioeconomic characteristics.

These relationships should be interpreted as predictive associations rather than causal effects.

An observed association between a predictor and GLP-1 use does not establish that the predictor causes medication use.

## Limitations

### Self-reported outcome

The target variable is based on survey responses. Reported medication use may therefore be affected by recall or reporting error.

### Cross-sectional data

The survey data do not establish temporal relationships between all predictors and GLP-1 use.

Consequently, the model should not be interpreted as a causal model of medication access or prescribing.

### Generalizability

The model is based on CDC Round 7 Rapid Surveys data and may not generalize to populations or time periods outside the survey.

### Survey design and prediction

The survey-weighted analysis improves population-level descriptive estimation, but the predictive model itself was developed using standard machine-learning methods.

The model should therefore not be interpreted as a fully survey-weighted population prediction model.

### Imputed panel variables

Some panel profile variables, including education-related variables, contain imputed values supplied as part of the CDC dataset.

### Small subgroups

Some race/ethnicity and other demographic categories contain relatively few observations. Performance estimates for these groups may therefore be unstable.

### Clinical use

The model is not intended to make prescribing, diagnosis, treatment, or insurance-coverage decisions.

## Reproducibility

The project stores data-processing, model-training, evaluation, robustness, subgroup, and survey-analysis scripts in the `src/` directory.

The trained final model is saved in:

`models/final_logistic_model.joblib`

Raw CDC data are intentionally excluded from version control.

Processed analysis outputs are stored in:

`data/processed/`