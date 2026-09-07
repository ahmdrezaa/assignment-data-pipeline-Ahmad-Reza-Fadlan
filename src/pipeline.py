from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "automobileEDA_dirty_training.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "automobileEDA_processed.csv"


def load_data(filepath: Path) -> pd.DataFrame:
    """Baca CSV lokal ke DataFrame."""
    df = pd.read_csv(filepath)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Periksa kondisi awal dataset."""
    print("\nDATA INSPECTION")

    print("\nLima baris pertama:")
    print(df.head())

    print("\nJumlah baris dan kolom:")
    print(df.shape)

    print("\nNama kolom:")
    print(df.columns.tolist())

    print("\nTipe data:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nJumlah duplicate records:")
    print(df.duplicated().sum())

    print("\nNilai unik kolom kategorikal:")
    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].unique())


def parse_transaction_date(value):
    """Ubah format tanggal menjadi datetime."""
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%b-%Y",
    ]

    for date_format in formats:
        parsed = pd.to_datetime(
            value,
            format=date_format,
            errors="coerce",
        )

        if pd.notna(parsed):
            return parsed

    return pd.NaT


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values + duplicates + format + noisy values."""
    df = df.copy()

    rows_before = len(df)
    missing_before = int(df.isna().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    df["transaction_date"] = df["transaction_date"].apply(
        parse_transaction_date
    )

    categorical_columns = [
        "make",
        "aspiration",
        "num-of-doors",
        "body-style",
        "drive-wheels",
        "engine-location",
        "engine-type",
        "num-of-cylinders",
        "fuel-system",
        "horsepower-binned",
    ]

    for column in categorical_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.lower()
            .str.strip()
        )

    df = df.drop_duplicates().reset_index(drop=True)

    for i in range(1, len(df) - 1):
        previous_make = df.at[i - 1, "make"]
        current_make = df.at[i, "make"]
        next_make = df.at[i + 1, "make"]

        if (
            pd.notna(previous_make)
            and pd.notna(next_make)
            and previous_make == next_make
            and (
                pd.isna(current_make)
                or current_make != previous_make
            )
        ):
            df.at[i, "make"] = previous_make

    if df["make"].isna().any():
        df["make"] = df["make"].fillna(
            df["make"].mode().iloc[0]
        )

    df["transaction_date"] = (
        df["transaction_date"]
        .interpolate(method="linear")
    )

    df["num-of-doors"] = df["num-of-doors"].fillna(
        df["num-of-doors"].mode().iloc[0]
    )

    df["stroke"] = df["stroke"].fillna(
        df["stroke"].median()
    )

    horsepower_group_median = (
        df.groupby("horsepower-binned")["horsepower"]
        .transform("median")
    )

    df["horsepower"] = (
        df["horsepower"]
        .fillna(horsepower_group_median)
        .fillna(df["horsepower"].median())
    )

    make_price_median = (
        df.groupby("make")["price"]
        .transform("median")
    )

    df["price"] = (
        df["price"]
        .fillna(make_price_median)
        .fillna(df["price"].median())
    )

    horsepower_bins = [
        -np.inf,
        101,
        155,
        np.inf,
    ]

    horsepower_labels = [
        "low",
        "medium",
        "high",
    ]

    derived_horsepower_bin = pd.cut(
        df["horsepower"],
        bins=horsepower_bins,
        labels=horsepower_labels,
    ).astype("string")

    df["horsepower-binned"] = (
        df["horsepower-binned"]
        .fillna(derived_horsepower_bin)
    )

    rows_after = len(df)
    missing_after = int(df.isna().sum().sum())
    duplicates_after = int(df.duplicated().sum())
    duplicates_removed = duplicates_before - duplicates_after

    print("\nDATA CLEANING")
    print(f"Jumlah data sebelum cleaning : {rows_before}")
    print(f"Jumlah data sesudah cleaning : {rows_after}")
    print(f"Missing values sebelum       : {missing_before}")
    print(f"Missing values sesudah       : {missing_after}")
    print(f"Duplicate records dihapus    : {duplicates_removed}")

    print("\nKolom yang mengalami cleaning:")
    print("- transaction_date")
    print("- make")
    print("- num-of-doors")
    print("- body-style")
    print("- drive-wheels")
    print("- fuel-system")
    print("- stroke")
    print("- horsepower")
    print("- price")
    print("- horsepower-binned")

    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Scaling + encoding."""
    df = df.copy()

    scaler = MinMaxScaler()

    scaled_values = scaler.fit_transform(
        df[["price", "horsepower"]]
    )

    df["price_norm"] = scaled_values[:, 0]
    df["horsepower_norm"] = scaled_values[:, 1]

    print("\nMIN-MAX SCALING")
    print(
        df[
            [
                "price",
                "price_norm",
                "horsepower",
                "horsepower_norm",
            ]
        ].head()
    )

    encoded_body_style = pd.get_dummies(
        df["body-style"],
        prefix="body",
        dtype=int,
    )

    df = pd.concat(
        [
            df,
            encoded_body_style,
        ],
        axis=1,
    )

    print("\nONE-HOT ENCODING")
    print(encoded_body_style.head())

    return df


def save_data(df: pd.DataFrame, filepath: Path) -> None:
    """Simpan processed dataset ke CSV."""
    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        filepath,
        index=False,
    )

    print(f"\nProcessed dataset disimpan ke: {filepath}")


def main() -> None:
    print("Mulai jalankan pipeline...")

    df = load_data(RAW_DATA_PATH)

    inspect_data(df)

    df = clean_data(df)

    df = transform_data(df)

    save_data(
        df,
        PROCESSED_DATA_PATH,
    )

    print("\nPipeline selesai")


if __name__ == "__main__":
    main()