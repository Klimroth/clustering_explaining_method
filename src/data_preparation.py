import os
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


import config


class DataPreparator:

    @staticmethod
    def apply_row_scaling(
        df: pd.DataFrame, feature_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, List[float | str]]]:
        
        output_df = df.copy()
        scaling_values: Dict[str, List[float | str]] = {"feature": [], "scaling": []}
        for feature in feature_columns:
            scaling_values["feature"].append(feature)
            scaling_values["scaling"].append(np.max(output_df[feature].to_list()))
        output_df[feature_columns] = output_df[feature_columns].apply(lambda x: x / x.max(), axis=0)

        return output_df, scaling_values

    @staticmethod
    def read_excel_sheet(
        path_to_excel: str, feature_columns: List[str]
    ) -> pd.DataFrame | None:
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
        upsampling_list: List[pd.DataFrame] = [df]
        for class_index, group in df.groupby(group_name):
            df_group_sample: pd.DataFrame = group.sample(
                max_group_size - len(group), replace=True
            )
            df_group_sample["oversampled"] = True
            upsampling_list.append(df_group_sample)
        return pd.concat(upsampling_list).reset_index(drop=True)

    @staticmethod
    def apply_imputation(
        df: pd.DataFrame, columns: List[str], use_config:bool=True, nn_imputation_k:int=10
        ) -> pd.DataFrame:
        if use_config:
            nn_imputation_k = config.NN_IMPUTATION_K
        knn_imputer: KNNImputer = KNNImputer(n_neighbors=nn_imputation_k)
        imputed_data = knn_imputer.fit_transform(df[columns])
        df_temp: pd.DataFrame = pd.DataFrame(imputed_data)
        df_temp.columns = columns
        df[columns] = df_temp[columns]
        return df

    @staticmethod
    def apply_distortion(
        df: pd.DataFrame, columns: List[str],
        use_config:bool=True, distortion_mean:float=0., distortion_std:float=0.001) -> pd.DataFrame:
        if use_config:
            distortion_mean = config.DISTORTION_MEAN
            distortion_std  = config.DISTORTION_STD

        dim_df: Tuple[int, int] = (len(df), len(columns))
        distortion: np.array = np.random.normal(
            loc=distortion_mean, scale=distortion_std, size=dim_df
        )
        df[columns] = df[columns].astype(np.float64)
        df.loc[:, columns] += distortion
        return df

    @staticmethod
    def get_observable_dataframe(
        use_config:bool=True, df:pd.DataFrame|None=None, distortion_mean:float=0., distortion_std:float=0.001,
        feature_columns:List[str]|None=None, nn_imputation_k:int=10, group_name:str='',
        ) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
        if use_config:
            feature_columns: List[str] = list(config.OBSERVABLE_FEATURE_NAMES.keys())
            df = DataPreparator.read_excel_sheet(
                config.INPUT_FILE_OBSERVABLE_FEATURES, feature_columns
            )
            group_name = config.GROUP_NAME
        else:
            if feature_columns is None:
                feature_columns: List[str] = list(df.columns)

        if df is None:
            return None, None

        df, scaling_information = DataPreparator.apply_row_scaling(df, feature_columns)
        df = DataPreparator.apply_imputation(
            df, feature_columns, use_config=use_config, nn_imputation_k=nn_imputation_k
        )
        df = DataPreparator.apply_oversampling(df, group_name)
        df = DataPreparator.apply_distortion(
            df, feature_columns,
            use_config=use_config, distortion_mean=distortion_mean, distortion_std=distortion_std
        )
        return df, pd.DataFrame(scaling_information)

    @staticmethod
    def get_explainable_dataframe(
        use_config:bool=True, df:pd.DataFrame|None=None,
        distortion_mean:float=0., distortion_std:float=0.001,
        feature_columns:List[str]|None=None
        ) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:

        if use_config:
            feature_columns: List[str] = list(config.EXPLAINING_FEATURE_NAMES.keys())
            df = DataPreparator.read_excel_sheet(
                config.INPUT_FILE_EXPLAINING_FEATURES, feature_columns
            )
        else:
            if feature_columns is None:
                feature_columns: List[str] = list(df.columns)

        if df is None:
            return None, None

        df, scaling_information = DataPreparator.apply_row_scaling(df, feature_columns)
        df = DataPreparator.apply_distortion(
            df, feature_columns,
            use_config=use_config, distortion_mean=distortion_mean, distortion_std=distortion_std
        )
        return df, pd.DataFrame(scaling_information)

    @staticmethod
    def not_all_existent(file_list: List[str]) -> bool:
        for filename in file_list:
            if not os.path.exists(filename):
                return True
        return False

    @staticmethod
    def prepare_data(
        use_config:bool=True,
        df_explainable:pd.DataFrame|None=None,
        df_observable:pd.DataFrame|None=None,
        distortion_mean:float=0., distortion_std:float=0.001,
        observed_features:List[str]|None=None,
        explaining_features:List[str]|None=None,
        nn_imputation_k:int=10,
        group_name:str='Index'
    )->None|dict:

        if use_config:
            save_folder: str = f"{config.OUTPUT_FOLDER_BASE}base_data/"
            group_name = config.GROUP_NAME
            required_files: List[str] = [
                f"{save_folder}{config.DATASET_NAME}_explainable_dataset_scaled.xlsx",
                f"{save_folder}{config.DATASET_NAME}_explainable_scaling_factors.xlsx",
                f"{save_folder}{config.DATASET_NAME}_observable_dataset_scaled.xlsx",
                f"{save_folder}{config.DATASET_NAME}_observable_scaling_factors.xlsx",
                f"{save_folder}{config.DATASET_NAME}_sample_size.xlsx",
            ]
            requirement = not config.USE_CACHED_DATASET or DataPreparator.not_all_existent(required_files)
            observed_features = list(config.OBSERVABLE_FEATURE_NAMES.keys())
        else:
            requirement = True

        if requirement:

            explainable_df, explainable_scaling = (
                DataPreparator.get_explainable_dataframe(
                    use_config=use_config, df=df_explainable,
                    distortion_mean=distortion_mean, distortion_std=distortion_std,
                    feature_columns=explaining_features
            )

            )
            observable_df, observable_scaling = (
                DataPreparator.get_observable_dataframe(
                    use_config=use_config, df=df_observable,
                    distortion_mean=distortion_mean, distortion_std=distortion_std,
                    nn_imputation_k=nn_imputation_k, group_name=group_name,
                    feature_columns=observed_features
                )
            )

            # Pandas seems to work differently
            # Depending on its version
            try:
                sample_size_df: pd.DataFrame = (
                    observable_df[observable_df["oversampled"] == False]
                    .groupby(group_name)
                    .count()[observed_features[0]]
                )
            except:
                sample_size_df: pd.DataFrame = (
                    observable_df[observable_df["oversampled"] is False]
                    .groupby(group_name)
                    .count()[observed_features[0]]
                )
            sample_size_df.columns = ["Sample size"]

            if use_config:

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

                explainable_df.to_excel(
                    f"{save_folder}{config.DATASET_NAME}_explainable_dataset_scaled.xlsx",
                    index=False,
                )
                explainable_scaling.to_excel(
                    f"{save_folder}{config.DATASET_NAME}_explainable_scaling_factors.xlsx",
                    index=False,
                )

                observable_df.to_excel(
                    f"{save_folder}{config.DATASET_NAME}_observable_dataset_scaled.xlsx",
                    index=False,
                )
                observable_scaling.to_excel(
                    f"{save_folder}{config.DATASET_NAME}_observable_scaling_factors.xlsx",
                    index=False,
                )
                sample_size_df.to_excel(
                    f"{save_folder}{config.DATASET_NAME}_sample_size.xlsx",
                    index=True,
                )

            else:

                result_dict = {
                    'explainable_df': explainable_df,
                    'explainable_scaling': explainable_scaling,
                    'observable_df': observable_df,
                    'observable_scaling': observable_scaling,
                    'sample_size_df': sample_size_df
                }

                return result_dict

        
