from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    import fasttext
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "fasttext is not installed. Install it with: pip install fasttext"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "xgboost is not installed. Install it with: pip install xgboost"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "clean_dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "fasttext_xgboost_hybrid_output"

RANDOM_STATE = 42
TEST_SIZE = 0.2
FASTTEXT_VAL_SIZE = 0.2
FASTTEXT_EPOCHS = [5]
OOF_FOLDS = 5

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
NUMERIC_COLUMNS = [
    "like_rt_ratio",
    "reply_rt_ratio",
    "word_count",
    "caps_ratio",
    "mentions",
    "URLs",
    "has_hashtag",
    "has_exclamation",
]
HYBRID_FEATURE_COLUMNS = [*NUMERIC_COLUMNS, "fasttext_prob"]


@dataclass
class FastTextEpochResult:
    epoch: int
    train_accuracy: float
    val_accuracy: float


def clean_text_for_fasttext(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"@\w+", " ", regex=True)
        .str.replace(r"http\S+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [LABEL_COLUMN, TEXT_COLUMN, *NUMERIC_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    validated = df.copy()
    validated[TEXT_COLUMN] = clean_text_for_fasttext(validated[TEXT_COLUMN])
    validated[LABEL_COLUMN] = pd.to_numeric(validated[LABEL_COLUMN], errors="coerce")

    for column in NUMERIC_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    validated = validated.dropna(subset=[LABEL_COLUMN]).copy()
    validated = validated[validated[LABEL_COLUMN].isin([0, 1])].copy()
    validated = validated[validated[TEXT_COLUMN].ne("")].copy()
    validated = validated[validated[TEXT_COLUMN].str.split().str.len().ge(3)].copy()
    validated[LABEL_COLUMN] = validated[LABEL_COLUMN].astype(int)

    return validated.reset_index(drop=True)


def to_fasttext_lines(df: pd.DataFrame) -> list[str]:
    text = clean_text_for_fasttext(df[TEXT_COLUMN])
    labels = "__label__" + df[LABEL_COLUMN].astype(int).astype(str)
    lines = (labels + " " + text).str.strip()
    return [line for line in lines.tolist() if line]


def write_fasttext_file(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def train_fasttext_model(train_file: Path, epoch: int) -> fasttext.FastText._FastText:
    return fasttext.train_supervised(
        input=str(train_file),
        epoch=epoch,
        lr=0.5,
        wordNgrams=2,
        minCount=2,
        bucket=200000,
        dim=50,
        loss="softmax",
        thread=1,
        verbose=0,
    )


def evaluate_fasttext_accuracy(
    model: fasttext.FastText._FastText,
    df: pd.DataFrame,
) -> float:
    if df.empty:
        return 0.0

    predictions = []
    for text in clean_text_for_fasttext(df[TEXT_COLUMN]):
        labels, _ = model.predict(text, k=1)
        predictions.append(int(labels[0].replace("__label__", "")))

    return float(accuracy_score(df[LABEL_COLUMN].to_numpy(), np.array(predictions)))


def evaluate_epochs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[list[FastTextEpochResult], int]:
    results: list[FastTextEpochResult] = []
    with tempfile.TemporaryDirectory(prefix="fasttext_epoch_selection_") as tmp_dir:
        train_path = Path(tmp_dir) / "fasttext_train.txt"
        write_fasttext_file(to_fasttext_lines(train_df), train_path)

        for epoch in FASTTEXT_EPOCHS:
            model = train_fasttext_model(train_path, epoch=epoch)
            results.append(
                FastTextEpochResult(
                    epoch=epoch,
                    train_accuracy=evaluate_fasttext_accuracy(model, train_df),
                    val_accuracy=evaluate_fasttext_accuracy(model, val_df),
                )
            )

    best_result = max(results, key=lambda item: (item.val_accuracy, -item.epoch))
    return results, best_result.epoch


def get_label1_probability(
    model: fasttext.FastText._FastText,
    text: str,
) -> float:
    labels, probabilities = model.predict(text, k=2)
    probability_map = dict(zip(labels, probabilities))
    return float(probability_map.get("__label__1", 0.0))


def generate_oof_fasttext_probabilities(
    train_df: pd.DataFrame,
    best_epoch: int,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probabilities = np.zeros(len(train_df), dtype=float)

    for fold_number, (fit_index, holdout_index) in enumerate(
        splitter.split(train_df, train_df[LABEL_COLUMN]),
        start=1,
    ):
        fold_train_df = train_df.iloc[fit_index].reset_index(drop=True)
        fold_holdout_df = train_df.iloc[holdout_index].reset_index(drop=True)

        with tempfile.TemporaryDirectory(prefix=f"fasttext_oof_fold_{fold_number}_") as tmp_dir:
            train_file = Path(tmp_dir) / "fold_train.txt"
            write_fasttext_file(to_fasttext_lines(fold_train_df), train_file)
            model = train_fasttext_model(train_file, epoch=best_epoch)

            holdout_probabilities = [
                get_label1_probability(model, text)
                for text in clean_text_for_fasttext(fold_holdout_df[TEXT_COLUMN])
            ]
            oof_probabilities[holdout_index] = np.array(holdout_probabilities, dtype=float)

    return oof_probabilities


def train_final_fasttext_model(
    train_df: pd.DataFrame,
    best_epoch: int,
    output_model_path: Path,
) -> fasttext.FastText._FastText:
    with tempfile.TemporaryDirectory(prefix="fasttext_final_train_") as tmp_dir:
        final_train_path = Path(tmp_dir) / "fasttext_full_train.txt"
        write_fasttext_file(to_fasttext_lines(train_df), final_train_path)
        model = train_fasttext_model(final_train_path, epoch=best_epoch)
    model.save_model(str(output_model_path))
    return model


def build_hybrid_features(
    df: pd.DataFrame,
    fasttext_probabilities: np.ndarray,
    medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    features = df[NUMERIC_COLUMNS].copy()
    features["fasttext_prob"] = fasttext_probabilities

    if medians is None:
        medians_series = features.median(numeric_only=True)
        medians = {column: float(value) for column, value in medians_series.items()}

    filled_features = features.fillna(pd.Series(medians))
    return filled_features[HYBRID_FEATURE_COLUMNS], medians


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    model = XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_hybrid_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions)),
        "threshold": 0.5,
        "sample_size": int(len(y_test)),
    }


def save_deployment_artifacts(
    model: XGBClassifier,
    best_epoch: int,
    medians: dict[str, float],
    metrics: dict[str, object],
    epoch_results: list[FastTextEpochResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "xgboost_model.json"))

    serializable_epoch_results = [
        {
            "epoch": item.epoch,
            "train_accuracy": item.train_accuracy,
            "validation_accuracy": item.val_accuracy,
        }
        for item in epoch_results
    ]

    (output_dir / "preprocessing.json").write_text(
        json.dumps(
            {
                "numeric_columns": NUMERIC_COLUMNS,
                "hybrid_feature_columns": HYBRID_FEATURE_COLUMNS,
                "feature_medians": medians,
                "best_fasttext_epoch": best_epoch,
                "positive_label": "FAKE",
                "negative_label": "REAL",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "hybrid_model": metrics,
                "fasttext_epoch_results": serializable_epoch_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = validate_dataset(df)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COLUMN],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    fasttext_train_df, fasttext_val_df = train_test_split(
        train_df,
        test_size=FASTTEXT_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df[LABEL_COLUMN],
    )
    fasttext_train_df = fasttext_train_df.reset_index(drop=True)
    fasttext_val_df = fasttext_val_df.reset_index(drop=True)

    epoch_results, best_epoch = evaluate_epochs(fasttext_train_df, fasttext_val_df)

    train_fasttext_prob = generate_oof_fasttext_probabilities(train_df, best_epoch=best_epoch)
    final_fasttext_model = train_final_fasttext_model(
        train_df,
        best_epoch=best_epoch,
        output_model_path=OUTPUT_DIR / "model.bin",
    )
    test_fasttext_prob = np.array(
        [
            get_label1_probability(final_fasttext_model, text)
            for text in clean_text_for_fasttext(test_df[TEXT_COLUMN])
        ],
        dtype=float,
    )

    X_train_hybrid, medians = build_hybrid_features(train_df, train_fasttext_prob)
    X_test_hybrid, _ = build_hybrid_features(test_df, test_fasttext_prob, medians=medians)

    y_train = train_df[LABEL_COLUMN].astype(int)
    y_test = test_df[LABEL_COLUMN].astype(int)

    hybrid_model = train_xgboost(X_train_hybrid, y_train)
    metrics = evaluate_hybrid_model(hybrid_model, X_test_hybrid, y_test)
    save_deployment_artifacts(
        model=hybrid_model,
        best_epoch=best_epoch,
        medians=medians,
        metrics=metrics,
        epoch_results=epoch_results,
        output_dir=OUTPUT_DIR,
    )

    print("Hybrid artifacts ready.")
    print(f"FastText model : {OUTPUT_DIR / 'model.bin'}")
    print(f"XGBoost model  : {OUTPUT_DIR / 'xgboost_model.json'}")
    print(f"Config         : {OUTPUT_DIR / 'preprocessing.json'}")
    print(f"Metrics        : {OUTPUT_DIR / 'metrics_summary.json'}")
    print(f"Accuracy       : {metrics['accuracy']:.4f}")
    print(f"F1 score       : {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
