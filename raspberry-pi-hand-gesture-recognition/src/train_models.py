"""Train three classifiers, compare them, and save the best model."""

import argparse
import csv
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate(name, model, x_test, y_test):
    predictions = model.predict(x_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_ROOT / "results" / "landmark_features.joblib",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = joblib.load(args.features)
    x_train, y_train = data["X_train"], data["y_train"]
    x_test, y_test = data["X_test"], data["y_test"]

    candidates = {
        "SVM (RBF)": SVC(kernel="rbf", C=10.0, probability=True),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42
        ),
    }

    rows = []
    trained_models = {}
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        trained_models[name] = model
        row = evaluate(name, model, x_test, y_test)
        rows.append(row)
        print(f"  accuracy={row['accuracy']:.4f}, f1={row['f1_score']:.4f}")

    results_path = PROJECT_ROOT / "results" / "model_comparison.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["accuracy"])
    model_path = PROJECT_ROOT / "models" / "best_landmark_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models[best["model"]], model_path)
    print(f"Best model: {best['model']} ({best['accuracy']:.2%})")
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
