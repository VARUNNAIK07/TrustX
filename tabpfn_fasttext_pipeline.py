from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect
import re
import tempfile
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split as sklearn_train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier

try:
    import fasttext
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fasttext is not installed. Install it with: pip install fasttext") from exc

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


DATA_PATH = Path(r"C:\Users\usrav\OneDrive\Desktop\dataset\twitter_analysis_dataset\clean_dataset.csv")
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "fasttext_tabpfn_output"

RANDOM_STATE = 42
TEST_SIZE = 0.2
OOF_FOLDS = 3
MAX_TABPFN_TRAIN_SAMPLES = 10000
FASTTEXT_VAL_SIZE = 0.2
FASTTEXT_EPOCH_CANDIDATES = [3]
FASTTEXT_DIM = 50
FASTTEXT_MIN_COUNT = 2
PERMUTATION_SAMPLE_SIZE = 1000
PERMUTATION_REPEATS = 2

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
FASTTEXT_PROB_COLUMN = "fasttext_probability"

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

FINAL_FEATURE_COLUMNS = [FASTTEXT_PROB_COLUMN, *NUMERIC_COLUMNS]


@dataclass
class PipelineArtifacts:
    fasttext_model: fasttext.FastText._FastText
    scaler: StandardScaler
    numeric_medians: pd.Series
    feature_medians: pd.Series
    classifier: TabPFNClassifier
    actual_device: str
    preferred_device: str
    device_reason: str


@dataclass
class FastTextEpochResult:
    epoch: int
    train_accuracy: float
    val_accuracy: float


def clean_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"http\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"@\w+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [TEXT_COLUMN, LABEL_COLUMN, *NUMERIC_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    validated = df.copy()
    validated[TEXT_COLUMN] = validated[TEXT_COLUMN].fillna("").astype(str).map(clean_text)
    validated[LABEL_COLUMN] = pd.to_numeric(validated[LABEL_COLUMN], errors="coerce")

    for column in NUMERIC_COLUMNS:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    validated = validated.dropna(subset=[LABEL_COLUMN]).copy()
    validated = validated[validated[LABEL_COLUMN].isin([0, 1])].copy()
    validated = validated[validated[TEXT_COLUMN].ne("")].copy()
    validated[LABEL_COLUMN] = validated[LABEL_COLUMN].astype(int)

    return validated.reset_index(drop=True)


def train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = sklearn_train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL_COLUMN],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def dataframe_to_fasttext_lines(df: pd.DataFrame) -> list[str]:
    labels = "__label__" + df[LABEL_COLUMN].astype(int).astype(str)
    lines = (labels + " " + df[TEXT_COLUMN]).str.strip()
    return [line for line in lines.tolist() if line]


def write_fasttext_file(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def train_fasttext_supervised(train_file: Path, epoch: int) -> fasttext.FastText._FastText:
    return fasttext.train_supervised(
        input=str(train_file),
        epoch=epoch,
        dim=FASTTEXT_DIM,
        minCount=FASTTEXT_MIN_COUNT,
        lr=0.5,
        wordNgrams=2,
        loss="softmax",
        bucket=200000,
        thread=4,
        verbose=0,
    )


def get_fasttext_probability(model: fasttext.FastText._FastText, text: str) -> float:
    labels, probabilities = model.predict(clean_text(text), k=2)
    probability_map = dict(zip(labels, probabilities))
    return float(probability_map.get("__label__1", 0.0))


def evaluate_fasttext_accuracy(
    model: fasttext.FastText._FastText,
    df: pd.DataFrame,
) -> float:
    if df.empty:
        return 0.0

    predictions = []
    for text in df[TEXT_COLUMN]:
        predictions.append(int(get_fasttext_probability(model, text) >= 0.5))

    return accuracy_score(df[LABEL_COLUMN].to_numpy(), np.array(predictions))


def evaluate_fasttext_epochs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epoch_candidates: list[int],
) -> tuple[list[FastTextEpochResult], int]:
    results: list[FastTextEpochResult] = []

    print(f"[FastText Epoch Eval] Starting on {len(train_df)} train rows and {len(val_df)} validation rows")
    with tempfile.TemporaryDirectory(prefix="fasttext_epoch_search_") as tmp_dir:
        train_file = Path(tmp_dir) / "fasttext_train.txt"
        write_fasttext_file(dataframe_to_fasttext_lines(train_df), train_file)

        for epoch in epoch_candidates:
            print(f"[FastText Epoch Eval] Training epoch={epoch} ...")
            model = train_fasttext_supervised(train_file, epoch=epoch)
            train_accuracy = evaluate_fasttext_accuracy(model, train_df)
            val_accuracy = evaluate_fasttext_accuracy(model, val_df)

            print(
                f"[FastText Epoch Eval] epoch={epoch} "
                f"| train_acc={train_accuracy:.4f} | val_acc={val_accuracy:.4f}"
            )
            results.append(
                FastTextEpochResult(
                    epoch=epoch,
                    train_accuracy=train_accuracy,
                    val_accuracy=val_accuracy,
                )
            )

    best_epoch = max(results, key=lambda item: (item.val_accuracy, -item.epoch)).epoch
    print(f"[FastText Epoch Eval] Best epoch selected: {best_epoch}")
    return results, best_epoch


def plot_fasttext_overfitting_curve(
    epoch_results: list[FastTextEpochResult],
    output_path: Path,
) -> None:
    epochs = [item.epoch for item in epoch_results]
    train_scores = [item.train_accuracy for item in epoch_results]
    val_scores = [item.val_accuracy for item in epoch_results]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_scores, marker="o", linewidth=2, label="Training Accuracy")
    plt.plot(epochs, val_scores, marker="o", linewidth=2, label="Validation Accuracy")
    plt.xlabel("FastText Epochs")
    plt.ylabel("Accuracy")
    plt.title("FastText Overfitting Curve")
    plt.xticks(epochs)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_oof_fasttext(train_df: pd.DataFrame, epoch: int) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probabilities = np.zeros(len(train_df), dtype=float)

    print(f"[OOF FastText] Starting OOF generation with {OOF_FOLDS} folds")
    start_time = time.perf_counter()

    for fold_number, (fit_idx, val_idx) in enumerate(
        splitter.split(train_df, train_df[LABEL_COLUMN]),
        start=1,
    ):
        fold_start = time.perf_counter()
        print(
            f"[OOF FastText] Fold {fold_number}/{OOF_FOLDS} "
            f"| train_rows={len(fit_idx)} | val_rows={len(val_idx)}"
        )

        fold_train_df = train_df.iloc[fit_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        with tempfile.TemporaryDirectory(prefix=f"tabpfn_ft_fold_{fold_number}_") as tmp_dir:
            train_file = Path(tmp_dir) / "fasttext_train.txt"
            write_fasttext_file(dataframe_to_fasttext_lines(fold_train_df), train_file)
            fold_model = train_fasttext_supervised(train_file, epoch=epoch)

            fold_val_probabilities = [
                get_fasttext_probability(fold_model, text)
                for text in fold_val_df[TEXT_COLUMN]
            ]
            oof_probabilities[val_idx] = np.array(fold_val_probabilities, dtype=float)

        print(f"[OOF FastText] Fold {fold_number}/{OOF_FOLDS} complete in {time.perf_counter() - fold_start:.2f} sec")

    print(f"[OOF FastText] All folds complete in {time.perf_counter() - start_time:.2f} sec")
    return oof_probabilities


def train_final_fasttext(
    train_df: pd.DataFrame,
    model_output_path: Path,
    epoch: int,
) -> fasttext.FastText._FastText:
    print(f"[Final FastText] Training final model with epoch={epoch} on {len(train_df)} rows")
    start_time = time.perf_counter()

    full_train_file = model_output_path.parent / "fasttext_full_train.txt"
    write_fasttext_file(dataframe_to_fasttext_lines(train_df), full_train_file)
    model = train_fasttext_supervised(full_train_file, epoch=epoch)
    model.save_model(str(model_output_path))

    print(f"[Final FastText] Complete in {time.perf_counter() - start_time:.2f} sec")
    return model


def build_features(
    df: pd.DataFrame,
    fasttext_probabilities: np.ndarray,
    scaler: StandardScaler | None = None,
    numeric_medians: pd.Series | None = None,
    feature_medians: pd.Series | None = None,
    fit: bool = False,
) -> tuple[np.ndarray, StandardScaler, pd.Series, pd.Series]:
    numeric_df = df[NUMERIC_COLUMNS].copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")

    if fit:
        numeric_medians = numeric_df.median(numeric_only=True)
    if numeric_medians is None:
        raise ValueError("numeric_medians must be provided when fit=False")

    numeric_df = numeric_df.fillna(numeric_medians)

    if fit:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(numeric_df)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        scaled_numeric = scaler.transform(numeric_df)

    feature_df = pd.DataFrame(scaled_numeric, columns=NUMERIC_COLUMNS, index=df.index)
    feature_df.insert(0, FASTTEXT_PROB_COLUMN, fasttext_probabilities)

    if fit:
        feature_medians = feature_df.median(numeric_only=True)
    elif feature_medians is None:
        raise ValueError("feature_medians must be provided when fit=False")

    feature_df = feature_df.fillna(feature_medians)

    return feature_df[FINAL_FEATURE_COLUMNS].to_numpy(dtype=np.float32), scaler, numeric_medians, feature_medians


def cap_training_size(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(X_train) <= MAX_TABPFN_TRAIN_SAMPLES:
        return X_train, y_train

    X_small, _, y_small, _ = sklearn_train_test_split(
        X_train,
        y_train,
        train_size=MAX_TABPFN_TRAIN_SAMPLES,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    return X_small, y_small


def resolve_tabpfn_device() -> tuple[str, str]:
    if torch is None:
        print("[CUDA Check] PyTorch not installed. CUDA unavailable.")
        return "cpu", "PyTorch is not installed, so CUDA cannot be used."

    cuda_available = torch.cuda.is_available()
    print(f"[CUDA Check] torch.cuda.is_available(): {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[CUDA Check] GPU detected: {gpu_name}")
        return "cuda", f"CUDA detected: {gpu_name}"

    print("[CUDA Check] No CUDA device available. Falling back to CPU.")
    return "cpu", "CUDA is not available. Falling back to CPU."


def build_tabpfn_classifier(preferred_device: str) -> tuple[TabPFNClassifier, str]:
    init_signature = inspect.signature(TabPFNClassifier)
    init_kwargs = {}

    if "device" in init_signature.parameters:
        init_kwargs["device"] = preferred_device
        actual_device = preferred_device
    else:
        actual_device = "cpu"

    classifier = TabPFNClassifier(**init_kwargs)
    return classifier, actual_device


def train_tabpfn(X_train: np.ndarray, y_train: np.ndarray) -> tuple[TabPFNClassifier, str, str, str]:
    preferred_device, device_reason = resolve_tabpfn_device()
    classifier, actual_device = build_tabpfn_classifier(preferred_device)

    print(f"[TabPFN] Requested device: {preferred_device}")
    print(f"[TabPFN] Running on: {actual_device}")
    print(f"[TabPFN] Training rows before capping: {len(X_train)}")

    X_train_capped, y_train_capped = cap_training_size(X_train, y_train)
    print(f"[TabPFN] Training rows after capping: {len(X_train_capped)}")

    start_time = time.perf_counter()
    classifier.fit(X_train_capped, y_train_capped)
    print(f"[TabPFN] Training complete in {time.perf_counter() - start_time:.2f} sec")
    print(f"[TabPFN] Actual device used: {actual_device}")
    print(f"[TabPFN] Model parameters on GPU: {actual_device == 'cuda' and preferred_device == 'cuda'}")

    return classifier, preferred_device, actual_device, device_reason


def evaluate_model(model: TabPFNClassifier, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    print("[Evaluation] Running test set predictions...")
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4, zero_division=0),
        "predictions": y_pred,
    }


def plot_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Fake", "Real"],
        cmap="Blues",
        colorbar=False,
    )
    disp.ax_.set_title("TabPFN Confusion Matrix")
    disp.ax_.set_xlabel("Predicted Label")
    disp.ax_.set_ylabel("True Label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_feature_importance_figure(
    model: TabPFNClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> None:
    print("[Feature Importance] Starting fast permutation importance...")

    if len(X_test) > PERMUTATION_SAMPLE_SIZE:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(len(X_test), size=PERMUTATION_SAMPLE_SIZE, replace=False)
        X_test_small = X_test[sample_idx]
        y_test_small = y_test[sample_idx]
        print(f"[Feature Importance] Using sampled test subset: {len(X_test_small)} rows")
    else:
        X_test_small = X_test
        y_test_small = y_test
        print(f"[Feature Importance] Using full test set: {len(X_test_small)} rows")

    importance = permutation_importance(
        model,
        X_test_small,
        y_test_small,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=-1,
    )

    sorted_idx = np.argsort(importance.importances_mean)[::-1]
    sorted_names = np.array(feature_names)[sorted_idx]
    sorted_means = importance.importances_mean[sorted_idx]
    sorted_stds = importance.importances_std[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_names)), sorted_means, yerr=sorted_stds, capsize=4)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right")
    plt.ylabel("Permutation Importance (F1 decrease)")
    plt.title("TabPFN Feature Importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[Feature Importance] Saved: {output_path}")


def extract_numeric_features_from_text(text: str, numeric_medians: pd.Series) -> pd.DataFrame:
    cleaned = clean_text(text)
    tokens = cleaned.split()
    alpha_count = sum(char.isalpha() for char in cleaned)
    uppercase_count = sum(char.isupper() for char in cleaned if char.isalpha())

    features = {
        "like_rt_ratio": float(numeric_medians["like_rt_ratio"]),
        "reply_rt_ratio": float(numeric_medians["reply_rt_ratio"]),
        "word_count": float(len(tokens)),
        "caps_ratio": float(uppercase_count / alpha_count) if alpha_count else 0.0,
        "mentions": float(len(re.findall(r"@\w+", str(text or "")))),
        "URLs": float(len(re.findall(r"http\S+|www\.\S+", str(text or "")))),
        "has_hashtag": float(int("#" in str(text or ""))),
        "has_exclamation": float(int("!" in str(text or ""))),
    }

    return pd.DataFrame([features], columns=NUMERIC_COLUMNS)


def predict_new_text(text: str, artifacts: PipelineArtifacts) -> dict[str, object]:
    cleaned_text = clean_text(text)
    single_row_numeric = extract_numeric_features_from_text(text, artifacts.numeric_medians)
    fasttext_probability = np.array(
        [get_fasttext_probability(artifacts.fasttext_model, cleaned_text)],
        dtype=np.float32,
    )

    single_feature_df = pd.DataFrame(
        {
            TEXT_COLUMN: [cleaned_text],
            **{column: single_row_numeric.iloc[0][column] for column in NUMERIC_COLUMNS},
        }
    )

    X_new, _, _, _ = build_features(
        df=single_feature_df,
        fasttext_probabilities=fasttext_probability,
        scaler=artifacts.scaler,
        numeric_medians=artifacts.numeric_medians,
        feature_medians=artifacts.feature_medians,
        fit=False,
    )

    predicted_label = int(artifacts.classifier.predict(X_new)[0])
    predicted_proba = float(artifacts.classifier.predict_proba(X_new)[0, 1])

    return {
        "cleaned_text": cleaned_text,
        "fasttext_probability": float(fasttext_probability[0]),
        "predicted_label": predicted_label,
        "predicted_label_name": "real" if predicted_label == 1 else "fake",
        "tabpfn_probability_class_1": predicted_proba,
    }


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("Model Evaluation\n")
        file.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        file.write(f"Precision: {metrics['precision']:.4f}\n")
        file.write(f"Recall   : {metrics['recall']:.4f}\n")
        file.write(f"F1 Score : {metrics['f1_score']:.4f}\n\n")
        file.write("Confusion Matrix\n")
        file.write(np.array2string(metrics["confusion_matrix"]))
        file.write("\n\nClassification Report\n")
        file.write(metrics["classification_report"])
        file.write("\n")


def main() -> None:
    total_start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Step 1/8] Loading dataset...")
    df = load_dataset()
    df = validate_dataset(df)
    print(f"[Step 1/8] Dataset loaded and validated: {len(df)} rows")

    print("[Step 2/8] Creating train/test split...")
    train_df, test_df = train_test_split(df)
    print(f"[Step 2/8] Split complete | train={len(train_df)} | test={len(test_df)}")

    print("[Step 3/8] Starting FastText epoch evaluation...")
    fasttext_train_df, fasttext_val_df = sklearn_train_test_split(
        train_df,
        test_size=FASTTEXT_VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df[LABEL_COLUMN],
    )
    fasttext_train_df = fasttext_train_df.reset_index(drop=True)
    fasttext_val_df = fasttext_val_df.reset_index(drop=True)

    epoch_results, best_fasttext_epoch = evaluate_fasttext_epochs(
        train_df=fasttext_train_df,
        val_df=fasttext_val_df,
        epoch_candidates=FASTTEXT_EPOCH_CANDIDATES,
    )
    plot_fasttext_overfitting_curve(epoch_results, OUTPUT_DIR / "fasttext_overfitting_curve.png")
    print("[Step 3/8] FastText epoch evaluation complete")

    y_train = train_df[LABEL_COLUMN].astype(int).to_numpy()
    y_test = test_df[LABEL_COLUMN].astype(int).to_numpy()

    print("[Step 4/8] Starting OOF FastText generation...")
    oof_start_time = time.perf_counter()
    train_oof_fasttext_prob = generate_oof_fasttext(train_df, epoch=best_fasttext_epoch)
    oof_elapsed = time.perf_counter() - oof_start_time
    print(f"[Step 4/8] OOF FastText generation complete in {oof_elapsed:.2f} sec")

    print("[Step 5/8] Training final FastText model...")
    final_fasttext_start = time.perf_counter()
    final_fasttext_model = train_final_fasttext(
        train_df=train_df,
        model_output_path=OUTPUT_DIR / "fasttext_model.bin",
        epoch=best_fasttext_epoch,
    )
    final_fasttext_elapsed = time.perf_counter() - final_fasttext_start
    print(f"[Step 5/8] Final FastText training complete in {final_fasttext_elapsed:.2f} sec")

    print("[Step 6/8] Building train/test features...")
    test_fasttext_prob = np.array(
        [get_fasttext_probability(final_fasttext_model, text) for text in test_df[TEXT_COLUMN]],
        dtype=np.float32,
    )

    X_train, scaler, numeric_medians, feature_medians = build_features(
        df=train_df,
        fasttext_probabilities=train_oof_fasttext_prob,
        fit=True,
    )
    X_test, _, _, _ = build_features(
        df=test_df,
        fasttext_probabilities=test_fasttext_prob,
        scaler=scaler,
        numeric_medians=numeric_medians,
        feature_medians=feature_medians,
        fit=False,
    )
    print(f"[Step 6/8] Feature building complete | X_train={X_train.shape} | X_test={X_test.shape}")

    print("[Step 7/8] Training TabPFN...")
    tabpfn_start = time.perf_counter()
    classifier, preferred_device, actual_device, device_reason = train_tabpfn(X_train, y_train)
    tabpfn_elapsed = time.perf_counter() - tabpfn_start
    print(f"[Step 7/8] TabPFN training complete in {tabpfn_elapsed:.2f} sec")

    print("[Step 8/8] Evaluating model and generating plots...")
    metrics = evaluate_model(classifier, X_test, y_test)
    save_metrics(metrics, OUTPUT_DIR / "metrics.txt")

    plot_confusion_matrix_figure(
        y_true=y_test,
        y_pred=metrics["predictions"],
        output_path=OUTPUT_DIR / "confusion_matrix.png",
    )
    plot_feature_importance_figure(
        model=classifier,
        X_test=X_test,
        y_test=y_test,
        feature_names=FINAL_FEATURE_COLUMNS,
        output_path=OUTPUT_DIR / "feature_importance.png",
    )
    print("[Step 8/8] Evaluation complete")

    artifacts = PipelineArtifacts(
        fasttext_model=final_fasttext_model,
        scaler=scaler,
        numeric_medians=numeric_medians,
        feature_medians=feature_medians,
        classifier=classifier,
        actual_device=actual_device,
        preferred_device=preferred_device,
        device_reason=device_reason,
    )

    total_elapsed = time.perf_counter() - total_start_time

    print(f"\nRows used              : {len(df)}")
    print(f"Training rows          : {len(train_df)}")
    print(f"Test rows              : {len(test_df)}")
    print(f"Best FastText epoch    : {best_fasttext_epoch}")
    print(f"FastText dim           : {FASTTEXT_DIM}")
    print(f"FastText minCount      : {FASTTEXT_MIN_COUNT}")
    print(f"TabPFN preferred device: {preferred_device}")
    print(f"TabPFN actual device   : {actual_device}")
    print(f"Device status          : {device_reason}")

    print("\nTiming Summary")
    print(f"OOF FastText time      : {oof_elapsed:.2f} sec")
    print(f"Final FastText time    : {final_fasttext_elapsed:.2f} sec")
    print(f"TabPFN training time   : {tabpfn_elapsed:.2f} sec")
    print(f"Total runtime          : {total_elapsed:.2f} sec")

    print("\nModel Evaluation")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    print("\nConfusion Matrix")
    print(metrics["confusion_matrix"])

    print("\nClassification Report")
    print(metrics["classification_report"])

    print(f"\nSaved FastText model   : {OUTPUT_DIR / 'fasttext_model.bin'}")
    print(f"Saved metrics          : {OUTPUT_DIR / 'metrics.txt'}")
    print(f"Saved plot             : {OUTPUT_DIR / 'fasttext_overfitting_curve.png'}")
    print(f"Saved plot             : {OUTPUT_DIR / 'confusion_matrix.png'}")
    print(f"Saved plot             : {OUTPUT_DIR / 'feature_importance.png'}")

    print("\nExample prediction")
    print(
        predict_new_text(
            "Breaking! Officials confirm the report after a full investigation.",
            artifacts,
        )
    )


if __name__ == "__main__":
    main()
