# Logistic Regression Notes

These notes accompany the final [Diabetes Prediction | Logistic Regression notebook](../notebooks/Diabetes_Prediction_Logistic_Regression.ipynb). They explain the modeling decisions in enough detail to reproduce the workflow and discuss it in an interview.

> **Scope:** An educational portfolio exercise, not a validated clinical screening or diagnostic system.

## 1. Why Logistic Regression?

Logistic Regression is a useful binary-classification baseline because it is relatively simple, fast to train, and interpretable. It combines the input features linearly, then uses the sigmoid function to estimate the positive-class probability:

`p = 1 / (1 + exp(-z))`, where `z = intercept + Σ(coefficient × feature)`.

The model estimates probabilities; a **decision threshold** converts those probabilities into predicted labels. Changing the threshold changes the labels and confusion matrix, but **not** the ROC-AUC of the underlying probability scores.

## 2. Dataset and Missing-Value Assumptions

- **768 records**, **8 numerical predictors**, and target `Outcome` (1 = diabetes, 0 = no diabetes).
- Dataset documentation describes women aged 21 or older of Pima Indian heritage. The CSV itself does not include those demographic columns, so results cannot be assumed to generalize to other populations.
- `Pregnancies = 0` is valid. Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing-value markers, a **modeling assumption**.
- `Insulin` has **48.7%** zeros; `SkinThickness` has **29.6%**. I keep the rows and use median imputation rather than discarding a large part of the dataset.

The exploratory correlation matrix uses cleaned values and pairwise complete observations. Missing pairs are omitted, so different correlations can use different sample sizes. EDA was not used to choose features or tune the model.

## 3. Train/Test Split and Leakage-Safe Pipeline

The stratified 80/20 split contains **614 training rows** and **154 test rows**, with `random_state=42`.

The pipeline is:

1. `SimpleImputer(strategy="median")`
2. `StandardScaler()`
3. `LogisticRegression(max_iter=1000, random_state=42)`

**Why this matters:** both median imputation and scaling learn parameters from data. If they are fitted before cross-validation, information from validation folds leaks into training. Putting them in the pipeline means each fold fits its own preprocessing only on that fold's training portion. The final model fits preprocessing on the training set and applies it unchanged to test data and new examples.

## 4. Baseline and Hyperparameter Tuning

I evaluated the baseline with **five-fold stratified cross-validation on the training set**. Its mean ROC-AUC is about **0.843**, with fold-to-fold standard deviation **0.019**. This is **mean ± standard deviation**, not a confidence interval.

`GridSearchCV` compared `model__C = [0.001, 0.01, 0.1, 1, 10, 100]`, scoring with ROC-AUC. Smaller `C` means stronger regularization. The notebook keeps the default **L2 penalty**; it does **not** search L1 versus L2 or use a nested GridSearchCV procedure.

- Best `C` from GridSearchCV: **0.1**.
- Best training-CV ROC-AUC: about **0.844**.
- Improvement over baseline: approximately **0.001** ROC-AUC.
- My **predefined training-only decision rule** requires a gain of at least **0.005** to switch models. The gain did not meet that bar, so the **baseline (`C=1`) remains the final model**.

The fourth decimal of CV scores can vary slightly across environments. The decision does not depend on that digit.

## 5. Threshold Selection on Training Data

A default threshold of **0.50** is convenient but not always aligned with the goal. For this exercise, I assumed missing a positive case mattered more than an unnecessary follow-up flag, and set a **training-OOF recall target of at least 80%**.

The notebook uses `cross_val_predict(..., method="predict_proba")` on **training data**. Each patient's OOF probability comes from a fold model that did not train on that patient. Thresholds from **0.05 to 0.95**, in steps of 0.01, are compared on those probabilities.

**Selection rule:** choose the **highest** threshold whose OOF recall is at least **0.80**, to limit false positives among the candidates meeting the goal.

| Threshold | Training-OOF recall | Training-OOF precision |
|---|---:|---:|
| 0.28 (selected) | 0.804 | 0.581 |
| 0.30 | 0.780 | 0.588 |
| 0.50 (default) | 0.575 | 0.759 |

The selected threshold is **0.28**. The 80% target is an educational assumption, **not** a clinically established requirement. Because the same OOF predictions help select the threshold, their scores are not an independent estimate of the entire threshold-selection process. The workflow is **not fully nested**.

## 6. Final Test Results

The baseline pipeline is fitted on the **614 training rows** and evaluated on **154 held-out test rows**.

| Metric | Threshold 0.50 | Threshold 0.28 |
|---|---:|---:|
| Accuracy | 0.708 | 0.727 |
| Precision | 0.600 | 0.577 |
| Recall / sensitivity | 0.500 | 0.833 |
| F1 | 0.545 | 0.682 |
| ROC-AUC | 0.813 | 0.813 |

At threshold **0.28**:

- **True negatives (TN): 67**
- **False positives (FP): 33**
- **False negatives (FN): 9**
- **True positives (TP): 45**

The model detects **45 of 54** positive cases and incorrectly flags **33 of 100** negative cases. At threshold 0.50, the confusion matrix is **TN=82, FP=18, FN=27, TP=27**.

**Important distinction:** positive-class recall (sensitivity) is `TP / (TP + FN) = 0.833`. Specificity is `TN / (TN + FP) = 0.670`. Negative predictive value (NPV) is `TN / (TN + FN) ≈ 0.882`. The 0.88 precision reported for the negative class corresponds to **NPV**, not specificity. NPV depends on prevalence.

The blue–green confusion-matrix palette represents **counts**, not correct versus incorrect predictions. Both panels share the same count scale.

## 7. Uncertainty and Generalization

The notebook uses **2,000 bootstrap resamples of the test set**, holding the fitted model and selected threshold fixed, to obtain approximate 95% percentile intervals:

| Metric | Point estimate | Approximate 95% interval |
|---|---:|---:|
| ROC-AUC | 0.813 | 0.740–0.873 |
| Recall | 0.833 | 0.730–0.929 |
| Precision | 0.577 | 0.468–0.684 |

Even the lower ROC-AUC bound is above 0.50 on this test set, but these intervals **do not include** retraining uncertainty, threshold reselection, or changes in population. The small number of test positives makes the uncertainty meaningful.

**Test-set limitation:** The current notebook does not use test data to fit preprocessing, tune `C`, or select the threshold. However, the same test split was inspected in earlier development iterations. These results are therefore not entirely independent of the full development history. A genuinely external dataset would be needed for a stronger generalization claim.

## 8. Coefficients, Odds Ratios, and Causality

With standardized features, coefficients are on a comparable scale. An odds ratio is `exp(coefficient)` for a **one-standard-deviation increase**, holding the other predictors constant.

- `Glucose`: coefficient about **1.183**, odds ratio about **3.263**.
- `BMI`: coefficient about **0.689**, odds ratio about **1.992**.
- `Pregnancies`, `DiabetesPedigreeFunction`, and `Age` also have positive fitted coefficients.
- `BloodPressure`, `SkinThickness`, and `Insulin` have coefficients close to zero in this model.

An odds ratio of 3.263 is **not** a 3.263-fold increase in probability. Nor does a coefficient prove a causal effect. Missingness and correlated features can affect coefficient stability; for example, `BMI` correlates with `SkinThickness` (about 0.65), and `Glucose` with `Insulin` (about 0.58).

The example prediction in the notebook returns a model-estimated probability of **61.0%**, but the example is a **false positive**. Because probability calibration was not evaluated, this is **not** a validated clinical-risk estimate.

## 9. What I Would Improve Next

- Seek external validation on a new population, and assess subgroup performance where the data permit it.
- Test missingness indicators and alternatives to median imputation.
- Compare `StandardScaler` with robust scaling under cross-validation to assess sensitivity to outliers.
- Evaluate probability calibration and consider clinically meaningful decision costs.
- Use repeated validation or a fully nested procedure if estimating the full selection process becomes important.

## 10. Interview-Ready Explanations

**Why a Pipeline?** It prevents leakage because every validation fold learns imputation and scaling only from its training portion. It also makes new-patient prediction consistent.

**Why not keep the tuned model?** GridSearchCV improved ROC-AUC by only about 0.001, below the preselected 0.005 gain requirement. More tuning did not provide compelling evidence of improvement.

**Why 0.28?** It was the highest candidate threshold reaching at least 80% recall on training OOF predictions. It was not selected from test outcomes.

**Did recall improve without a cost?** No. Test recall rose from 0.500 to 0.833, while false positives rose from 18 to 33 and precision fell from 0.600 to 0.577.

**Can this model be used clinically?** No. It needs external validation, probability calibration, population-level checks, and real clinical input about the costs of false negatives and false positives.

**Main lesson:** A credible ML portfolio project should show not only what the model predicts, but also how its evaluation avoids leakage, why its threshold was chosen, and what the results cannot establish.
