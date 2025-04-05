from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import logging

logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, sample_size=0.1, random_state=42):
        self.model = XGBRegressor(n_jobs=-1)
        self.imputer = SimpleImputer(strategy='mean')
        self.sample_size = sample_size  # Fraction of data to use for training
        self.random_state = random_state

    def __call__(self, train_data, input_data, target_columns):
        # train data is a dataframe of features and target columns
        # input data is a dataframe of features
        # target columns is a list of target columns

        # Drop non-numeric columns, including timestamps, from train_data and input_data
        train_data_numeric = train_data.select_dtypes(include=['number'])
        input_data_numeric = input_data.select_dtypes(include=['number'])

        df_result = pd.DataFrame(index=input_data.index, columns=target_columns)
        df_result = df_result.fillna(0.0).infer_objects(copy=False)

        for target_column in tqdm(target_columns, desc='Training XGBoost models'):
            try:
                # Check if target column exists in train data
                if target_column not in train_data_numeric.columns:
                    logger.warning(f"Target column {target_column} not found in training data. Skipping.")
                    df_result[target_column] = 0.0
                    continue
                
                # Extract the target column from train dataset
                y = train_data_numeric[target_column].values
                
                # Skip if y is empty
                if len(y) == 0:
                    logger.warning(f"No data available for target column {target_column}. Skipping.")
                    df_result[target_column] = 0.0
                    continue
                
                # Create feature matrix X by dropping the target columns from train_data
                feature_columns = [col for col in train_data_numeric.columns if col not in target_columns]
                X_df = train_data_numeric[feature_columns]
                
                # Filter out columns with all NaN values or zero variance
                non_nan_cols = X_df.columns[~X_df.isna().all()]
                # Further filter to ensure we only keep columns with at least one non-missing value
                valid_feature_cols = [col for col in non_nan_cols if not X_df[col].isna().all()]
                
                # Skip if no valid features are available
                if not valid_feature_cols:
                    logger.warning(f"No valid features available for target column {target_column}. Skipping.")
                    df_result[target_column] = 0.0
                    continue
                
                # Use only valid features
                X = X_df[valid_feature_cols].values
                
                # Impute missing values using sklearn's SimpleImputer for both X and y
                y = self.imputer.fit_transform(y.reshape(-1, 1)).flatten()
                X = self.imputer.fit_transform(X)

                # Sample the data to reduce training time (if sample_size < 1)
                if self.sample_size < 1.0 and len(X) > 1 and len(y) > 1:
                    # Calculate actual sample size - ensure at least 1 sample
                    n_samples = max(1, int(len(X) * self.sample_size))
                    
                    # Use simple random sampling instead of train_test_split to avoid issues
                    indices = np.random.RandomState(self.random_state).choice(
                        len(X), size=n_samples, replace=False
                    )
                    X_sampled = X[indices]
                    y_sampled = y[indices]
                else:
                    X_sampled, y_sampled = X, y
                
                # Verify dimensions match before fitting
                if len(X_sampled) != len(y_sampled) or len(X_sampled) == 0 or len(y_sampled) == 0:
                    logger.warning(f"Dimension mismatch or empty arrays for {target_column}. X: {len(X_sampled)}, y: {len(y_sampled)}. Skipping.")
                    df_result[target_column] = 0.0
                    continue
                    
                # Fit the XGBoost model on the sampled data
                self.model.fit(X_sampled, y_sampled)

                # Prepare the input data for prediction (using the same valid features)
                X_input = input_data_numeric[valid_feature_cols].values
                
                # Skip if input data has no valid features
                if X_input.size == 0:
                    logger.warning(f"No valid input features for target column {target_column}. Skipping.")
                    df_result[target_column] = 0.0
                    continue
                    
                X_input = self.imputer.transform(X_input)

                # Predict the target column for the input data
                predictions = self.model.predict(X_input)
                df_result[target_column] = predictions
                
            except Exception as e:
                logger.error(f"Error processing target column {target_column}: {str(e)}")
                # In case of error, fill with zeros
                df_result[target_column] = 0.0

        return df_result


