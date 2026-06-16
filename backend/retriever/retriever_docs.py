documents = [

    # ── 0 ── Dataset first look
    "When you receive a new dataset, check five things before any modelling: "
    "(1) shape — the rows-to-columns ratio matters for model choice; "
    "(2) target distribution — is it balanced, skewed, or bimodal? "
    "(3) missing value percentage per column — above 50% usually means drop; "
    "(4) dtype correctness — numbers stored as strings, or dates stored as objects; "
    "(5) duplicate rows — duplicates in training data silently inflate every metric.",

    # ── 1 ── Identifying the task type from the target
    "Before choosing a model, always inspect the target column first. "
    "If it is continuous and unbounded, use regression metrics (MAE, RMSE, R²). "
    "If it has exactly two unique values (0/1, True/False, yes/no), it is binary classification — "
    "do not run regression on it even if the values look numeric. "
    "If it has 3–20 unique string or integer categories, it is multiclass classification. "
    "If it has hundreds of unique numeric values but is clearly bounded (e.g. ratings 1–5), "
    "consider both regression and ordinal classification before deciding.",

    # ── 2 ── ID column detection and removal
    "Columns that are unique identifiers (IDs, UUIDs, order numbers) must be removed before "
    "training — they have no predictive signal but a tree model will happily memorise them "
    "and appear to perform perfectly on training data. "
    "Detect ID columns by checking: unique count equals row count, dtype is integer or string, "
    "and column name contains words like 'id', 'key', 'code', 'index', or 'number'. "
    "Even if the name doesn't hint at it, a column with 100% unique values is almost always an identifier.",

    # ── 3 ── Data leakage — definition and most common form
    "Data leakage means the model receives information during training that it would not have "
    "at prediction time, causing unrealistically high validation scores that collapse in production. "
    "The most common form is target leakage: a feature that is computed after or directly from "
    "the target (e.g. 'total_paid' predicting 'churned', where churned customers paid nothing). "
    "A strong warning sign is a feature with near-perfect correlation (>0.95) with the target — "
    "legitimate features rarely achieve this. "
    "Always ask: 'Would this feature be available at the moment I need to make this prediction?'",

    # ── 4 ── Train/test split for time-series data
    "Never use random train/test split on time-series or any dataset with a temporal ordering. "
    "Random splitting allows future information to leak into the training set, making the model "
    "look better than it really is. "
    "Always split by time: train on the earlier period, validate on the later period. "
    "For cross-validation on time data, use TimeSeriesSplit from sklearn, which creates "
    "expanding or sliding window folds that respect temporal order. "
    "A model trained on 2023 data should never 'see' any row from 2024 during training.",

    # ── 5 ── Class imbalance — when it matters and what to do
    "Class imbalance matters when the minority class is the one you actually care about "
    "(e.g. fraud detection, rare disease diagnosis). "
    "Do not resample blindly: first check whether your metric is imbalance-aware "
    "(F1, ROC-AUC, and precision-recall are; raw accuracy is not). "
    "For moderate imbalance (5–20% minority), try class_weight='balanced' in the model before "
    "any resampling — it is simpler and often equally effective. "
    "For severe imbalance (<1% minority), SMOTE oversampling or undersampling the majority "
    "class can help, but always apply them only inside cross-validation folds, never before splitting.",

    # ── 6 ── Missing value strategy by type
    "The right strategy for missing values depends on why they are missing. "
    "If missing completely at random (MCAR), simple mean/median imputation works. "
    "If missing not at random (MNAR) — e.g. patients with extreme values refused the test — "
    "imputation introduces bias; add a binary indicator column 'feature_was_missing' and impute "
    "the value separately so the model can learn the missingness pattern. "
    "For tree-based models (XGBoost, LightGBM, CatBoost), you often do not need to impute at all — "
    "they handle NaN natively by learning the best split direction for missing values.",

    # ── 7 ── High-cardinality categorical encoding
    "When a categorical column has more than 20–30 unique values (city, product SKU, user agent), "
    "one-hot encoding creates too many sparse columns and usually hurts tree models. "
    "Use target encoding instead: replace each category with the mean of the target for that category. "
    "Critical: always compute target encoding inside cross-validation folds — "
    "computing it on the full training set before splitting leaks target information and inflates scores. "
    "For rare categories (appearing fewer than 5 times), group them into a single 'Other' bucket "
    "before encoding to avoid noisy estimates.",

    # ── 8 ── Ordinal vs one-hot encoding
    "Use ordinal (integer) encoding when the categories have a natural order and that order "
    "matters for the prediction: e.g. 'low/medium/high', 'bronze/silver/gold', education levels. "
    "Use one-hot encoding when categories are nominal — no meaningful order exists: "
    "e.g. city names, product categories, colours. "
    "Applying ordinal encoding to nominal categories forces the model to treat the arbitrary "
    "integer assignment as meaningful, which can hurt performance. "
    "For tree-based models, the distinction matters less because trees split on thresholds, "
    "but for linear models and neural networks it is critical.",

    # ── 9 ── Date and time feature extraction
    "A raw datetime column is useless to most models — extract features from it instead. "
    "Useful components to derive: year, month, day of week (0=Monday), hour, is_weekend (binary), "
    "is_holiday (requires a holiday calendar), and days_since_reference (a numeric trend feature). "
    "Cyclical features like hour and day of week should be encoded with sine/cosine pairs "
    "(sin(2π·hour/24), cos(2π·hour/24)) so the model understands that hour 23 is close to hour 0. "
    "After extraction, drop the original datetime column — most models cannot process it directly.",

    # ── 10 ── Log transformation — when and how
    "Apply log transformation to a numeric feature when its distribution is heavily right-skewed "
    "(long tail of large values) and you are using a linear model or neural network. "
    "Tree-based models (Random Forest, XGBoost, LightGBM) do not benefit from log transformation "
    "because their splits are invariant to monotonic transformations. "
    "Use np.log1p (log of 1+x) instead of np.log to safely handle zero values. "
    "Also apply log transformation to the target variable in regression if it is skewed — "
    "remember to exponentiate predictions back (np.expm1) when evaluating.",

    # ── 11 ── Outlier handling strategies
    "Before removing outliers, determine whether they are measurement errors or genuine extreme cases. "
    "Genuine extreme values (e.g. a billionaire's income in a dataset of people) carry real signal "
    "and should not be removed. "
    "For linear models sensitive to outliers, consider capping (clipping) values at the 1st and 99th "
    "percentiles rather than removing rows. "
    "Tree-based models are naturally robust to outliers in features because they split by rank. "
    "Outliers in the target variable are more dangerous — a single extreme target value can dominate "
    "the loss function in regression; log-transforming the target often mitigates this.",

    # ── 12 ── Feature correlation and multicollinearity
    "Highly correlated features (correlation > 0.95) are almost always redundant — "
    "one can be dropped without any loss of information. "
    "Multicollinearity (multiple correlated features) does not hurt tree-based models in terms "
    "of prediction quality, but it makes feature importance scores unreliable — importance gets "
    "split arbitrarily between correlated columns. "
    "For linear models and logistic regression, multicollinearity inflates coefficient variance "
    "and makes the model unstable; use L2 (Ridge) regularisation or remove correlated features. "
    "A practical threshold: if removing a feature does not change validation metric by more than "
    "0.001, it can safely be dropped.",

    # ── 13 ── Which models need feature scaling
    "Feature scaling (normalisation or standardisation) is required by models that compute "
    "distances or use gradient descent: k-NN, SVM, logistic regression, linear regression, "
    "neural networks, and PCA. "
    "Tree-based models — Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost — "
    "are completely invariant to monotonic feature transformations and do not benefit from scaling. "
    "A common mistake is scaling features before a Random Forest and wasting preprocessing time. "
    "When in doubt, always scale if your pipeline includes any distance-based or linear model.",

    # ── 14 ── Model selection: when to use XGBoost vs LightGBM vs CatBoost
    "XGBoost, LightGBM, and CatBoost all implement gradient boosting but have different strengths. "
    "Use LightGBM when speed matters most or the dataset is very large (>1M rows) — "
    "it uses histogram-based splits and leaf-wise tree growth that is significantly faster. "
    "Use CatBoost when the dataset has many categorical features — it handles them natively "
    "without manual encoding, using ordered target statistics to prevent leakage. "
    "Use XGBoost when you need the most mature ecosystem, best sklearn integration, "
    "or are working in an environment where LightGBM/CatBoost are not available. "
    "All three usually outperform Random Forest on tabular data if tuned properly.",

    # ── 15 ── Starting with a baseline model
    "Always build the simplest possible baseline before any complex model. "
    "For regression: predict the mean of the target for every row — this gives you a floor; "
    "any model that cannot beat this is broken or leaking. "
    "For classification: predict the majority class for every row, then compute your metric. "
    "A good baseline reveals how hard the problem actually is and prevents over-engineering. "
    "If a linear regression already achieves 0.92 R², there is no reason to spend days tuning XGBoost.",

    # ── 16 ── Overfitting diagnosis
    "Overfitting is confirmed when training metric is significantly better than validation metric. "
    "A gap of more than 5–10 percentage points between train and validation score is a red flag. "
    "Common causes: too many features relative to rows, a model with too high capacity (deep trees), "
    "or data leakage that only affects the training set. "
    "Fixes in order of simplicity: increase training data, reduce model complexity (max_depth, "
    "min_samples_leaf), add regularisation (L1/L2, dropout in neural nets), or reduce feature count "
    "by dropping low-importance features.",

    # ── 17 ── Choosing the right cross-validation strategy
    "K-fold cross-validation (typically k=5 or k=10) is the default choice for most tabular datasets. "
    "Use Stratified K-Fold for classification tasks — it ensures each fold has the same class "
    "distribution as the full dataset, which is critical for imbalanced datasets. "
    "Use TimeSeriesSplit for any data with temporal ordering — never shuffle time-series data. "
    "Use GroupKFold when rows are not independent (e.g. multiple rows per patient, per user, "
    "per store) — splitting randomly would leak information between train and validation.",

    # ── 18 ── Feature importance and feature selection
    "Tree-based models provide built-in feature importance, but use it carefully. "
    "Impurity-based importance (default in sklearn) is biased toward high-cardinality features "
    "and correlated features — it can rank a random ID column as highly important. "
    "Permutation importance (sklearn.inspection.permutation_importance) is more reliable: "
    "it measures the actual drop in metric when a feature's values are shuffled. "
    "A practical selection strategy: train the full model, compute permutation importance, "
    "then iteratively remove the lowest-importance features while monitoring validation metric.",

    # ── 19 ── Hyperparameter tuning order for boosting models
    "When tuning gradient boosting models, tune parameters in this order for efficiency. "
    "First: n_estimators (number of trees) — use early stopping with a validation set "
    "to find the optimal value automatically; this is the most impactful parameter. "
    "Second: max_depth and min_child_weight (or min_samples_leaf) — these control overfitting. "
    "Third: learning_rate — lower is better but requires more trees; set it last. "
    "Fourth: subsample and colsample_bytree — stochastic sampling often helps generalisation. "
    "Do not grid-search all parameters simultaneously — the search space is too large.",

    # ── 20 ── Early stopping in boosting
    "Always use early stopping when training XGBoost, LightGBM, or CatBoost. "
    "Early stopping monitors a validation metric after each tree and stops training when "
    "the metric has not improved for a set number of rounds (early_stopping_rounds=50 is typical). "
    "This prevents overfitting without manually tuning n_estimators, and speeds up training. "
    "Pass a separate eval_set that the model never trains on — do not use the training set for "
    "early stopping, as the training metric always improves monotonically and will never trigger stopping.",

    # ── 21 ── Validation set contamination — the silent killer
    "Validation set contamination occurs when information from validation data is used during "
    "any preprocessing step performed before the train/validation split. "
    "Common mistakes: fitting a StandardScaler on the full dataset before splitting, "
    "computing target encoding on the full training set, or selecting features based on "
    "correlation with the target computed across the full dataset. "
    "The fix is to use sklearn Pipeline: fit all preprocessing steps only on training data, "
    "then transform validation and test data using the fitted transformers. "
    "A model that appears to perform well but dramatically underperforms in production is "
    "almost always suffering from some form of validation contamination.",

    # ── 22 ── sklearn Pipeline — why always use it
    "A sklearn Pipeline chains preprocessing steps and a model into a single object that can "
    "be fitted, validated, and deployed without risk of data leakage between steps. "
    "When you call pipeline.fit(X_train, y_train), every transformer is fitted only on X_train. "
    "When you call pipeline.predict(X_val), transformers use the statistics from training — "
    "the validation set never influences any fitted parameter. "
    "Pipelines also simplify cross-validation: pass the pipeline directly to cross_val_score "
    "and leakage prevention is automatic across all folds.",

    # ── 23 ── Dealing with rare categories in test data (unseen labels)
    "When a categorical column in the test set contains values not seen during training "
    "(unseen categories), one-hot encoding will either crash or produce wrong-shaped arrays. "
    "Strategies: (1) use handle_unknown='ignore' in sklearn's OneHotEncoder to fill unknown "
    "categories with all zeros; (2) group rare training categories into 'Other' before "
    "encoding so test values have a bucket to fall into; "
    "(3) use CatBoost or LightGBM native categorical handling — they handle unseen values gracefully. "
    "Always check for unseen categories when deploying a model trained on historical data.",

    # ── 24 ── Metric selection for regression
    "MAE (mean absolute error) is the right metric when large errors are not disproportionately "
    "worse than small ones, and when the data contains outliers you do not want to overweight. "
    "RMSE (root mean squared error) heavily penalises large errors — use it when a prediction "
    "that is 10 units off is much worse than ten predictions that are 1 unit off. "
    "R² (coefficient of determination) measures the proportion of variance explained and is "
    "useful for communicating model quality to non-technical stakeholders, but is not suitable "
    "as a loss function because it can be negative for very bad models. "
    "MAPE (mean absolute percentage error) is interpretable but undefined when the target is zero.",

    # ── 25 ── Metric selection for classification
    "For balanced classification problems, accuracy is a fine starting metric. "
    "For imbalanced problems, use F1-score (harmonic mean of precision and recall) for the minority "
    "class, or ROC-AUC which measures discrimination across all classification thresholds. "
    "Precision and recall measure complementary things: precision is 'of all my positive predictions, "
    "how many were correct?', recall is 'of all real positives, how many did I find?'. "
    "If missing a positive is very costly (e.g. cancer diagnosis), optimise for recall. "
    "If false positives are very costly (e.g. fraud alerts causing customer friction), optimise for precision.",

    # ── 26 ── Feature engineering for geospatial data
    "Raw latitude and longitude columns are not directly useful as model features "
    "because Euclidean distance between lat/lon pairs does not correspond to real geographic distance. "
    "Useful derived features: Haversine distance from a fixed reference point (city centre, "
    "nearest store, headquarters), cluster assignment from k-means on coordinates, "
    "and administrative region (city, district) obtained by reverse geocoding. "
    "If multiple location points exist per row (origin and destination), their Haversine distance "
    "and bearing angle are often the most predictive features.",

    # ── 27 ── Stacking and blending — when it helps
    "Stacking (training a meta-model on the out-of-fold predictions of base models) consistently "
    "improves performance in ML competitions but offers diminishing returns in production. "
    "It helps most when base models make different types of errors — e.g. a tree model and a "
    "linear model disagree on different subgroups of the data. "
    "Blend diverse models rather than similar ones: stacking five gradient boosting variants "
    "adds little over a single well-tuned model. "
    "The meta-model should be simple (logistic regression or linear regression) to avoid "
    "overfitting the stacking layer on the validation fold predictions.",

    # ── 28 ── Sparse data and high-dimensional features
    "When the feature matrix is sparse (most values are zero) — common with one-hot encoded "
    "high-cardinality columns or text features — use scipy sparse matrices rather than dense "
    "numpy arrays to avoid memory exhaustion. "
    "Linear models (logistic regression, Ridge) work well on sparse high-dimensional data and "
    "are often competitive with or better than tree models in this regime. "
    "Tree-based models can be slow on very high-dimensional sparse data — LightGBM handles "
    "it better than XGBoost due to its histogram binning approach.",

    # ── 29 ── Diagnosing a model that is too good to be true
    "A model with near-perfect validation scores (AUC > 0.99, R² > 0.99) should immediately "
    "trigger a leakage investigation rather than celebration. "
    "Check for: (1) a feature that directly encodes the target; "
    "(2) a timestamp or sequential ID that encodes the split — the model learns 'early rows are "
    "class 0, late rows are class 1'; "
    "(3) duplicate rows across train and validation sets; "
    "(4) preprocessing steps fitted on the full dataset before splitting. "
    "Inspect the top-3 features by importance — if any are conceptually derived from the target, "
    "you have leakage.",

    # ── 30 ── Working with wide datasets (many columns, few rows)
    "When columns outnumber rows (p > n), standard models overfit severely. "
    "Start with aggressive feature selection: remove near-zero-variance features, "
    "then remove features with correlation > 0.95 with another feature. "
    "Use regularised models (Ridge, Lasso, ElasticNet) which penalise model complexity — "
    "they are specifically designed for the p > n regime. "
    "Cross-validation is especially critical here: with few rows, the variance of your metric "
    "estimate is high — use k=10 folds or repeated k-fold.",

    # ── 31 ── Encoding strategies for tree vs linear models
    "The best encoding strategy depends on the model type, not just the data. "
    "For tree-based models: ordinal (integer) encoding works fine for most categoricals "
    "because trees split on thresholds and do not assume numeric meaning; "
    "one-hot encoding is unnecessary and increases dimensionality. "
    "For linear models and neural networks: one-hot encoding is required for nominal categories "
    "because the model would treat integer codes as ordered magnitudes. "
    "CatBoost is the exception: pass raw categorical columns directly and it handles encoding "
    "internally using ordered target statistics that prevent leakage.",

    # ── 32 ── Practical train size vs model complexity trade-off
    "Model complexity should be proportional to training set size. "
    "With fewer than 1000 rows, simple models (logistic regression, Ridge, shallow trees) "
    "usually outperform complex ones because deep models overfit the small sample. "
    "With 10,000–100,000 rows, gradient boosting (XGBoost, LightGBM) is the first choice "
    "for tabular data — it consistently wins in this range. "
    "With more than 1M rows, neural networks become competitive with boosting on tabular data, "
    "and LightGBM is preferred over XGBoost due to its speed advantage at scale.",

    # ── 33 ── When neural networks beat gradient boosting on tabular data
    "For most tabular datasets, gradient boosting outperforms neural networks without extensive "
    "architecture search and tuning. "
    "Neural networks tend to win on tabular data when: the dataset is very large (>1M rows), "
    "there are complex interactions between many features that trees cannot capture efficiently, "
    "or the task involves embeddings (e.g. entity representations, user/item IDs). "
    "TabNet and other transformer-based tabular models have shown promise but rarely outperform "
    "well-tuned LightGBM without significant engineering effort. "
    "A practical rule: start with gradient boosting; switch to neural networks only if it "
    "plateaus and the dataset is large enough to support them.",

    # ── 34 ── Interpreting and communicating results
    "When presenting model results to non-technical stakeholders, avoid raw metric names. "
    "Instead of 'our model achieves MAE of 150', say 'our predictions are off by an average "
    "of £150, compared to £420 for the naive baseline — a 64% reduction in error'. "
    "For classification, a confusion matrix is more intuitive than a single accuracy number "
    "because it shows where the model fails specifically. "
    "SHAP values are the most effective tool for explaining individual predictions: "
    "they show exactly how much each feature contributed to moving the prediction up or down "
    "from the average.",

]

# ── Evaluation data ──────────────────────────────────────────────────────────
# Every relevant_doc index is verified against the content above.

evaluation_data = [
    {
        "query": "what to check first when I get a new dataset",
        "relevant_docs": [0]
    },
    {
        "query": "how to determine if the task is classification or regression",
        "relevant_docs": [1]
    },
    {
        "query": "how to detect and remove id columns",
        "relevant_docs": [2]
    },
    {
        "query": "what is data leakage and how to find it",
        "relevant_docs": [3]
    },
    {
        "query": "how to split time series data for training",
        "relevant_docs": [4]
    },
    {
        "query": "how to handle class imbalance",
        "relevant_docs": [5]
    },
    {
        "query": "how to deal with missing values",
        "relevant_docs": [6]
    },
    {
        "query": "how to encode high cardinality categorical features",
        "relevant_docs": [7]
    },
    {
        "query": "when to use ordinal encoding vs one-hot encoding",
        "relevant_docs": [8]
    },
    {
        "query": "how to extract features from a date column",
        "relevant_docs": [9]
    },
    {
        "query": "when to apply log transformation",
        "relevant_docs": [10]
    },
    {
        "query": "how to handle outliers in a dataset",
        "relevant_docs": [11]
    },
    {
        "query": "correlated features and multicollinearity",
        "relevant_docs": [12]
    },
    {
        "query": "which models need feature scaling",
        "relevant_docs": [13]
    },
    {
        "query": "xgboost vs lightgbm vs catboost when to use each",
        "relevant_docs": [14]
    },
    {
        "query": "how to build a baseline model",
        "relevant_docs": [15]
    },
    {
        "query": "how to detect overfitting",
        "relevant_docs": [16]
    },
    {
        "query": "which cross validation strategy to use",
        "relevant_docs": [17]
    },
    {
        "query": "how to select the most important features",
        "relevant_docs": [18]
    },
    {
        "query": "how to tune gradient boosting hyperparameters",
        "relevant_docs": [19]
    },
    {
        "query": "what is early stopping and how to use it",
        "relevant_docs": [20]
    },
    {
        "query": "validation set contamination and how to prevent it",
        "relevant_docs": [21]
    },
    {
        "query": "why use sklearn pipeline",
        "relevant_docs": [22]
    },
    {
        "query": "unseen categories in test data",
        "relevant_docs": [23]
    },
    {
        "query": "which regression metric should I use",
        "relevant_docs": [24]
    },
    {
        "query": "which classification metric should I use",
        "relevant_docs": [25]
    },
    {
        "query": "how to handle latitude longitude features",
        "relevant_docs": [26]
    },
    {
        "query": "when does model stacking help",
        "relevant_docs": [27]
    },
    {
        "query": "how to work with sparse high dimensional data",
        "relevant_docs": [28]
    },
    {
        "query": "my model is too accurate something is wrong",
        "relevant_docs": [29]
    },
    {
        "query": "more columns than rows what to do",
        "relevant_docs": [30]
    },
    {
        "query": "best encoding for tree models vs linear models",
        "relevant_docs": [31]
    },
    {
        "query": "which model to choose based on dataset size",
        "relevant_docs": [32]
    },
    {
        "query": "when do neural networks beat gradient boosting on tabular data",
        "relevant_docs": [33]
    },
    {
        "query": "how to explain model results to non-technical people",
        "relevant_docs": [34]
    },
    # Multi-doc queries
    {
        "query": "how to prevent overfitting in a boosting model",
        "relevant_docs": [16, 19, 20]
    },
    {
        "query": "feature engineering for a new dataset",
        "relevant_docs": [0, 9, 10, 11]
    },
    {
        "query": "train test split best practices",
        "relevant_docs": [4, 17, 21]
    },
    {
        "query": "how to choose between xgboost lightgbm and random forest",
        "relevant_docs": [14, 32]
    },
    {
        "query": "categorical feature encoding strategies",
        "relevant_docs": [7, 8, 23, 31]
    },
    {
        "query": "model is perfect on training data but bad on test data",
        "relevant_docs": [16, 21, 29]
    },
    {
        "query": "how to evaluate a classification model",
        "relevant_docs": [17, 25]
    },
    {
        "query": "target encoding vs one-hot encoding",
        "relevant_docs": [7, 8, 31]
    },
    {
        "query": "what to do with a time series dataset",
        "relevant_docs": [4, 9, 17]
    },
    {
        "query": "how to improve model performance",
        "relevant_docs": [15, 16, 18, 19]
    },
]

# Self-verification
if __name__ == "__main__":
    print(f"Documents: {len(documents)}")
    print(f"Evaluation queries: {len(evaluation_data)}")
    errors = []
    all_indices = set(range(len(documents)))
    for item in evaluation_data:
        for idx in item["relevant_docs"]:
            if idx >= len(documents):
                errors.append(f"Query '{item['query']}': index {idx} out of range")
    if errors:
        print("\nINDEX ERRORS:")
        for e in errors:
            print(" ", e)
    else:
        print("All indices valid.")

    # Confirm no doc is completely unreachable in evaluation
    covered = set(i for item in evaluation_data for i in item["relevant_docs"])
    uncovered = all_indices - covered
    if uncovered:
        print(f"Docs never marked relevant: {sorted(uncovered)}")
        for i in sorted(uncovered):
            print(f"  [{i}] {documents[i][:60]}...")
    else:
        print("All documents referenced in evaluation data.")
