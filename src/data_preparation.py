### observable features
# read data
# impute data
# over-sample by group (add row: oversampled True/False)
# re-scale row-wise -> save sacling factor per row to plot correctly later
# add gaussian noise to each entry, entries are [0,1]-valued
# save as excel:
# - scaling factors
# - sample size per group (not over-sampled)
# - data sheet

### explaining features
# read data
# re-scale row-wise -> save scaling factor
# save as excel -> scaling factors and scaled sheet

import os
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


import config
class DataPreparator:


    @staticmethod
    def apply_row_scaling(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
        scaling_values: Dict[str, float] = {}
        for feature in feature_columns:
            scaling_values[feature] = np.max(df[feature].to_list())
        df = df.apply(lambda x: x/x.max(), axis=1)

        return df, scaling_values

    @staticmethod
    def read_excel_sheet(path_to_excel: str, feature_columns: List[str]) -> pd.DataFrame|None:
        try:
            df = pd.read_excel(path_to_excel)
            columns: List[str] = [config.GROUP_NAME] + feature_columns
            df = df[columns]
        except Exception:
            print(f"ERROR: invalid file {config.INPUT_FILE_EXPLAINING_FEATURES}.")
            return None
        return df
    @staticmethod
    def apply_oversampling(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        max_group_size: int = df[group_name].value_counts().max()
        df["oversampled"] = False
        upsampling_list: List[df.DataFrame] = [df]
        for class_index, group in df.groupby(group_name):
            df_group_sample: pd.DataFrame = group.sample(max_group_size - len(group), replace=True)
            df_group_sample["oversampled"] = True
            upsampling_list.append(df_group_sample)
        return pd.concat(upsampling_list).reset_index(drop=True)

    @staticmethod
    def apply_imputation(df: pd.DataFrame, columns: List[str] ) -> pd.DataFrame:
        knn_imputer: KNNImputer = KNNImputer(n_neighbors=config.NN_IMPUTATION_K)
        imputed_data = knn_imputer.fit_transform(df[columns])
        df_temp: pd.DataFrame = pd.DataFrame(imputed_data)
        df_temp.columns = columns
        df[columns] = df_temp[columns]
        return df

    @staticmethod
    def apply_distortion(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        dim_df: Tuple[int, int] = (len(df), len(columns))
        distortion: np.array = np.random.normal(loc=config.DISTORTION_MEAN, scale=config.DISTORTION_STD, size=dim_df)
        df[columns].iloc[:] += distortion
        return df

    @staticmethod
    def get_observable_dataframe() -> Tuple[pd.DataFrame|None, pd.DataFrame|None]:
        feature_columns: List[str] = list(config.OBSERVABLE_FEATURE_NAMES.keys())
        df = DataPreparator.read_excel_sheet(config.INPUT_FILE_OBSERVABLE_FEATURES, feature_columns)

        if df is None:
            return None, None

        df, scaling_information = DataPreparator.apply_row_scaling(df, feature_columns)
        df = DataPreparator.apply_imputation(df, feature_columns)
        df = DataPreparator.apply_oversampling(df, config.GROUP_NAME)
        df = DataPreparator.apply_distortion(df, feature_columns)
        return df, pd.DataFrame(scaling_information)
    @staticmethod
    def get_explainable_dataframe() -> Tuple[pd.DataFrame|None, pd.DataFrame|None]:
        feature_columns: List[str] = list(config.EXPLAINING_FEATURE_NAMES.keys())
        df = DataPreparator.read_excel_sheet(config.INPUT_FILE_EXPLAINING_FEATURES, feature_columns)

        if df is None:
            return None, None

        df, scaling_information = DataPreparator.apply_row_scaling(df, feature_columns)
        df = DataPreparator.apply_distortion(df, feature_columns)
        return df, pd.DataFrame(scaling_information)
