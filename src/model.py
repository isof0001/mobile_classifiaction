"""Classify mobile phones into 4 price ranges with logistic regression."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"
PREDICTIONS_DIR = ROOT / "predictions"

TARGET = "price_range"
BINARY_FEATURES = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
SELECTED_FEATURES = ["ram", "battery_power", "px_height", "px_width"]


def mutual_information(df: pd.DataFrame) -> pd.Series:
    """Mutual information between each feature and the target."""
    X = df.drop(columns=TARGET)
    discrete = X.columns.isin(BINARY_FEATURES)
    mi = mutual_info_classif(X, df[TARGET], discrete_features=discrete, random_state=0)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def plot_mutual_information(mi: pd.Series, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    mi.plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set(xlabel=f"Mutual information with {TARGET}", title="Feature relevance")
    fig.tight_layout()
    fig.savefig(out / "mutual_information.png", dpi=150)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(out / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (held-out test set)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test.csv", help="unlabeled data to predict")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-features", action="store_true", help="use all 20 features instead of the 4 selected")
    parser.add_argument("--show", action="store_true", help="display plots in a window")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    IMAGES_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(args.train)
    print(f"Loaded {len(df)} rows, {df.isnull().sum().sum()} missing values\n")

    mi = mutual_information(df)
    print("Mutual information:\n", mi.round(4), "\n")
    plot_mutual_information(mi, IMAGES_DIR)
    plot_correlation(df, IMAGES_DIR)

    features = [c for c in df.columns if c != TARGET] if args.all_features else SELECTED_FEATURES
    print(f"Features used: {features}\n")

    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[TARGET], test_size=args.test_size, random_state=args.seed, stratify=df[TARGET]
    )

    # Scaler inside the pipeline: it is re-fitted on each CV training fold only.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))

    cv_auc = cross_val_score(model, x_train, y_train, cv=5, scoring="roc_auc_ovr")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    print(f"CV AUC (5-fold, train set): {cv_auc.round(4)}  mean={cv_auc.mean():.4f} std={cv_auc.std():.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test AUC (OvR): {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}\n")
    print(classification_report(y_test, y_pred, digits=3))
    plot_confusion(y_test, y_pred, IMAGES_DIR)

    # Predict the unlabeled test file with a model re-fitted on all labeled data.
    if args.test.exists():
        test_df = pd.read_csv(args.test)
        final_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
        final_model.fit(df[features], df[TARGET])
        out = pd.DataFrame({"id": test_df["id"], TARGET: final_model.predict(test_df[features])})
        out_path = PREDICTIONS_DIR / "test_predictions.csv"
        out.to_csv(out_path, index=False)
        print(f"Predictions for {len(out)} phones saved to {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
