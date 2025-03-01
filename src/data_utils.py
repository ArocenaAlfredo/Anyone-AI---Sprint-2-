import os
from typing import Tuple

import gdown
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download from GDrive all the needed datasets for the project.

    Returns:
        app_train : pd.DataFrame
            Training dataset

        app_test : pd.DataFrame
            Test dataset

        columns_description : pd.DataFrame
            Extra dataframe with detailed description about dataset features
    """
    # Download HomeCredit_columns_description.csv
    if not os.path.exists(config.DATASET_DESCRIPTION):
        gdown.download(
            config.DATASET_DESCRIPTION_URL, config.DATASET_DESCRIPTION, quiet=False
        )

    # Download application_test_aai.csv
    if not os.path.exists(config.DATASET_TEST):
        gdown.download(config.DATASET_TEST_URL, config.DATASET_TEST, quiet=False)

    # Download application_train_aai.csv
    if not os.path.exists(config.DATASET_TRAIN):
        gdown.download(config.DATASET_TRAIN_URL, config.DATASET_TRAIN, quiet=False)

    app_train = pd.read_csv(config.DATASET_TRAIN)
    app_test = pd.read_csv(config.DATASET_TEST)
    columns_description = pd.read_csv(config.DATASET_DESCRIPTION_URL)

    return app_train, app_test, columns_description



def get_feature_target(
    app_train: pd.DataFrame, app_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).
    """

    # Assign to X_train all the columns from app_train except "TARGET"
    X_train = app_train.drop(columns=["TARGET"])

    # Assign to y_train the "TARGET" column
    y_train = app_train["TARGET"]

    # Remove "TARGET" from X_test if it exists
    if "TARGET" in app_test.columns:
        X_test = app_test.drop(columns=["TARGET"])
    else:
        X_test = app_test  # Keep it as is if TARGET is not present

    # 🔥 Crear y_test con el mismo número de filas que app_test pero con valores NaN
    y_test = pd.Series([float("nan")] * app_test.shape[0], index=app_test.index)

    return X_train, y_train, X_test, y_test







def get_train_val_sets(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the training dataset into train and validation sets.
    """

    # Ensure inputs are not None
    if X_train is None or y_train is None:
        raise ValueError("X_train or y_train is None. Check data loading.")

    # Split into 80% training and 20% validation
    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True
    )

    # Convert y_train_new and y_val to Series if they are DataFrames
    if isinstance(y_train_new, pd.DataFrame):
        y_train_new = y_train_new.squeeze()
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.squeeze()

    # Debugging prints
    print(" X_train_new shape: ok", X_train_new.shape)
    print(" y_train_new shape: ok", y_train_new.shape)
    print(" X_val shape: ok", X_val.shape)
    print(" y_val shape: ok", y_val.shape)

    return X_train_new, X_val, y_train_new, y_val
