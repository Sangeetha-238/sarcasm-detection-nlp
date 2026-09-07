import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean_data(path):
    """
    Load the sarcasm dataset and apply basic cleaning.
    """
    df = pd.read_json(path, lines=True)

    df = df[["headline", "is_sarcastic"]].dropna()
    df = df.drop_duplicates(subset="headline")
    df = df.reset_index(drop=True)

    return df


def split_data(df, random_state=42):
    """
    Split the dataset into train, validation, and test sets.
    Uses a 70/15/15 stratified split.
    """
    X = df["headline"]
    y = df["is_sarcastic"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test