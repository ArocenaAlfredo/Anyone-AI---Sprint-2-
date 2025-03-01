from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder



def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses data for modeling by handling missing values, encoding categorical variables,
    correcting anomalies, and scaling features.

    Arguments:
        train_df : pd.DataFrame
            Training dataset
        val_df : pd.DataFrame
            Validation dataset
        test_df : pd.DataFrame
            Test dataset

    Returns:
        train : np.ndarray
            Processed training dataset
        val : np.ndarray
            Processed validation dataset
        test : np.ndarray
            Processed test dataset
    """

    # Print shape of input data
    print(" Input train data shape: ", train_df.shape)
    print(" Input val data shape: ", val_df.shape)
    print(" Input test data shape: ", test_df.shape, "\n")

    # Make copies of the dataframes to avoid modifying the originals
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    #  Correct outliers in `DAYS_EMPLOYED`
    for df in [working_train_df, working_val_df, working_test_df]:
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    #  Encode categorical variables
    categorical_cols = working_train_df.select_dtypes(include=["object"]).columns

    # Separate binary and multi-category columns
    binary_cols = [col for col in categorical_cols if working_train_df[col].nunique() == 2]
    one_hot_cols = [col for col in categorical_cols if working_train_df[col].nunique() > 2]

    # Apply binary encoding with OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    working_train_df[binary_cols] = ordinal_encoder.fit_transform(working_train_df[binary_cols])
    working_val_df[binary_cols] = ordinal_encoder.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ordinal_encoder.transform(working_test_df[binary_cols])

    # Apply one-hot encoding with OneHotEncoder
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_encoded = one_hot_encoder.fit_transform(working_train_df[one_hot_cols])
    val_encoded = one_hot_encoder.transform(working_val_df[one_hot_cols])
    test_encoded = one_hot_encoder.transform(working_test_df[one_hot_cols])

    # Convert encoded arrays to DataFrames
    train_encoded_df = pd.DataFrame(train_encoded, columns=one_hot_encoder.get_feature_names_out())
    val_encoded_df = pd.DataFrame(val_encoded, columns=one_hot_encoder.get_feature_names_out())
    test_encoded_df = pd.DataFrame(test_encoded, columns=one_hot_encoder.get_feature_names_out())

    # Drop original categorical columns and concatenate the encoded data
    working_train_df = working_train_df.drop(columns=one_hot_cols).reset_index(drop=True)
    working_val_df = working_val_df.drop(columns=one_hot_cols).reset_index(drop=True)
    working_test_df = working_test_df.drop(columns=one_hot_cols).reset_index(drop=True)

    working_train_df = pd.concat([working_train_df, train_encoded_df], axis=1)
    working_val_df = pd.concat([working_val_df, val_encoded_df], axis=1)
    working_test_df = pd.concat([working_test_df, test_encoded_df], axis=1)

    #  Handle missing values using median imputation
    imputer = SimpleImputer(strategy="median")
    working_train_df = pd.DataFrame(imputer.fit_transform(working_train_df), columns=working_train_df.columns)
    working_val_df = pd.DataFrame(imputer.transform(working_val_df), columns=working_val_df.columns)
    working_test_df = pd.DataFrame(imputer.transform(working_test_df), columns=working_test_df.columns)

    #  Scale features using Min-Max Scaler
    scaler = MinMaxScaler()
    working_train_df = scaler.fit_transform(working_train_df)
    working_val_df = scaler.transform(working_val_df)
    working_test_df = scaler.transform(working_test_df)

    return working_train_df, working_val_df, working_test_df

