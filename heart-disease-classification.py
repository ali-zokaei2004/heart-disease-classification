import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def calculate_metrics(model, X_train, X_test, y_train, y_test):
    """Calculate evaluation metrics for train/test sets + cross-validation."""
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Base metrics
    metrics = {
        "Accuracy Train": accuracy_score(y_train, y_pred_train),
        "Accuracy Test": accuracy_score(y_test, y_pred_test),
        "Precision Test": precision_score(y_test, y_pred_test),
        "Recall Test": recall_score(y_test, y_pred_test),
        "F1-Score Test": f1_score(y_test, y_pred_test),
    }

    # ROC-AUC (only if probability prediction is available)
    if hasattr(model, "predict_proba"):
        y_prob_test = model.predict_proba(X_test)[:, 1]
        metrics["ROC-AUC Test"] = roc_auc_score(y_test, y_prob_test)
    else:
        metrics["ROC-AUC Test"] = None  # some models (like SVM w/o prob) may not support

    # Cross-validation (on train set only to avoid data leakage)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    metrics["CV Accuracy"] = cv_scores.mean()

    return metrics, confusion_matrix(y_test, y_pred_test)


# Load dataset
data = pd.read_csv(
    "/Users/alizokaei/heart-disease-classification/heart.csv"
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("target", axis=1),
    data["target"],
    test_size=0.2,
    random_state=42,
    stratify=data["target"],
)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}

# Train, predict, and evaluate each model
results = []
conf_matrices = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    metrics, cm = calculate_metrics(model, X_train, X_test, y_train, y_test)
    metrics["Model"] = model_name
    results.append(metrics)
    conf_matrices[model_name] = cm

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# ==== Visualization ====

# Bar chart for model comparison
plt.figure(figsize=(12, 6))
sns.barplot(
    data=results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
    x="Model", y="Score", hue="Metric"
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Confusion matrices heatmaps
for model_name, cm in conf_matrices.items():
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()