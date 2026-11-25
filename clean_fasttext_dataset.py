import argparse
import csv
import re
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "label",
    "text",
    "like_rt_ratio",
    "reply_rt_ratio",
    "word_count",
    "caps_ratio",
    "mentions",
    "URLs",
    "has_hashtag",
    "has_exclamation",
]


def clean_tweet_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value)

    # Remove stray outer quotes/spaces
    text = text.strip().strip('"').strip("'")

    # Remove incorrect FastText label prefix like _label_0 or _label_1
    text = re.sub(r"^\s*_+label_+[01]\b\s*", "", text, flags=re.IGNORECASE)

    # Remove escaped and real newlines/carriage returns
    text = text.replace("\\n", " ")
    text = text.replace("\\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Remove backslashes
    text = text.replace("\\", " ")

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def load_dataset(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        input_path,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        keep_default_na=True,
        on_bad_lines="skip",
    )


def build_outputs(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()

    if "fasttext_format" not in working_df.columns:
        raise KeyError("Input CSV must contain a 'fasttext_format' column.")
    if "label" not in working_df.columns:
        raise KeyError("Input CSV must contain a 'label' column.")

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in working_df.columns and col != "text"]
    if missing_columns:
        raise KeyError(f"Input CSV is missing required columns: {missing_columns}")

    # Extract and clean tweet text
    working_df["text"] = working_df["fasttext_format"].apply(clean_tweet_text)

    # Recompute word count based on cleaned text
    working_df["word_count"] = working_df["text"].str.split().str.len()

    # Normalize label
    working_df["label"] = pd.to_numeric(working_df["label"], errors="coerce")

    # Remove empty / too-short / invalid-label rows
    cleaned_df = working_df[
        working_df["text"].ne("")
        & working_df["word_count"].ge(3)
        & working_df["label"].isin([0, 1])
    ].copy()

    cleaned_df["label"] = cleaned_df["label"].astype(int)

    # Guarantee required columns exist
    for col in EXPECTED_COLUMNS:
        if col not in cleaned_df.columns:
            cleaned_df[col] = pd.NA

    # Keep exact requested order
    cleaned_df = cleaned_df[EXPECTED_COLUMNS].reset_index(drop=True)

    return cleaned_df


def write_fasttext_file(df: pd.DataFrame, output_path: Path) -> None:
    lines = "__label__" + df["label"].astype(str) + " " + df["text"]

    # Final safety cleanup for FastText format
    lines = lines.str.replace(r"\s+", " ", regex=True).str.strip()

    # Remove empty lines if any slipped through
    lines = lines[lines.ne("")]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean fake news CSV and generate FastText files.")
    parser.add_argument(
        "--input",
        default=r"C:\Users\usrav\OneDrive\Desktop\dataset\twitter_analysis dataset\final_fasttext_features_dataset.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where outputs will be saved",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(input_path)
    rows_before = len(raw_df)

    cleaned_df = build_outputs(raw_df)
    rows_after = len(cleaned_df)

    clean_csv_path = output_dir / "clean_dataset.csv"
    train_txt_path = output_dir / "train.txt"

    # Save clean CSV
    cleaned_df.to_csv(clean_csv_path, index=False, encoding="utf-8")

    # Save FastText training file
    write_fasttext_file(cleaned_df, train_txt_path)

    print(f"Number of rows before cleaning: {rows_before}")
    print(f"Number of rows after cleaning: {rows_after}")
    print(f"Saved clean CSV to: {clean_csv_path.resolve()}")
    print(f"Saved FastText training file to: {train_txt_path.resolve()}")


if __name__ == "__main__":
    main()
