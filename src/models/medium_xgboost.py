from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class XGBoostModel:
    def __init__(self, sample_size=0.3, random_state=42):
        self.model = XGBRegressor(n_jobs=-1)
        self.sample_size = sample_size  # Fraction of data to use for training
        self.random_state = random_state

    def _engineer_features(self, df, country_code):
        """
        Add engineered features to the dataframe.
        
        Args:
            df: The input dataframe
            country_code: Either 'IT' or 'ES' to filter country-specific features
        """
        # Create a copy to avoid modifying the original dataframe
        df_engineered = df.copy()
        
        # Process datetime features if DATETIME column exists
        if 'DATETIME' in df.columns:
            # Convert DATETIME to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['DATETIME']):
                df_engineered['DATETIME'] = pd.to_datetime(df['DATETIME'])
            
            # Extract time-based features
            df_engineered['hour'] = df_engineered['DATETIME'].dt.hour
            df_engineered['day_of_week'] = df_engineered['DATETIME'].dt.dayofweek  # 0=Monday, 6=Sunday
            df_engineered['month'] = df_engineered['DATETIME'].dt.month
            df_engineered['quarter'] = df_engineered['DATETIME'].dt.quarter
            df_engineered['is_weekend'] = df_engineered['day_of_week'].isin([5, 6]).astype(int)
            
            # Hour cyclical encoding (to capture the cyclical nature of hours)
            df_engineered['hour_sin'] = np.sin(2 * np.pi * df_engineered['hour']/24)
            df_engineered['hour_cos'] = np.cos(2 * np.pi * df_engineered['hour']/24)
            
            # Day of week cyclical encoding
            df_engineered['day_of_week_sin'] = np.sin(2 * np.pi * df_engineered['day_of_week']/7)
            df_engineered['day_of_week_cos'] = np.cos(2 * np.pi * df_engineered['day_of_week']/7)
            
            # Season calculation (Northern Hemisphere)
            df_engineered['season'] = ((df_engineered['month'] % 12 + 3) // 3).astype(int)  # 1=Spring, 2=Summer, 3=Fall, 4=Winter
            
            # Specific time periods
            df_engineered['is_business_hours'] = ((df_engineered['hour'] >= 9) & 
                                                (df_engineered['hour'] <= 17) & 
                                                (df_engineered['is_weekend'] == 0)).astype(int)
        
        # Feature combinations and transformations - only for the specific country
        if country_code == 'ES' and 'TMP_ES' in df.columns:
            # Temperature squared (captures non-linear relationship with energy)
            df_engineered['TMP_squared'] = df_engineered['TMP_ES'] ** 2
            
            # Heating and cooling degree features (assuming 18°C is the base)
            df_engineered['heating_degree'] = np.maximum(18 - df_engineered['TMP_ES'], 0)
            df_engineered['cooling_degree'] = np.maximum(df_engineered['TMP_ES'] - 18, 0)
            
            # Solar photovoltaic features
            if 'SPV_ES' in df.columns:
                # Log transformation for SPV (often has a log relationship with generation)
                df_engineered['SPV_log'] = np.log1p(np.maximum(df_engineered['SPV_ES'], 0))
                
                # Interaction terms between weather and time features
                if 'hour' in df_engineered.columns:
                    df_engineered['hour_SPV'] = df_engineered['hour'] * df_engineered['SPV_ES']
                    
            # Holiday interaction features
            if 'is_holiday_ES' in df.columns and 'is_weekend' in df_engineered.columns:
                df_engineered['holiday_or_weekend'] = ((df_engineered['is_holiday_ES'] == 1) | 
                                                    (df_engineered['is_weekend'] == 1)).astype(int)
                
        elif country_code == 'IT' and 'TMP_IT' in df.columns:
            df_engineered['TMP_squared'] = df_engineered['TMP_IT'] ** 2
            df_engineered['heating_degree'] = np.maximum(18 - df_engineered['TMP_IT'], 0)
            df_engineered['cooling_degree'] = np.maximum(df_engineered['TMP_IT'] - 18, 0)
            
            # Solar photovoltaic features
            if 'SPV_IT' in df.columns:
                df_engineered['SPV_log'] = np.log1p(np.maximum(df_engineered['SPV_IT'], 0))
                
                # Interaction terms between weather and time features
                if 'hour' in df_engineered.columns:
                    df_engineered['hour_SPV'] = df_engineered['hour'] * df_engineered['SPV_IT']
                    
            # Holiday interaction features
            if 'is_holiday_IT' in df.columns and 'is_weekend' in df_engineered.columns:
                df_engineered['holiday_or_weekend'] = ((df_engineered['is_holiday_IT'] == 1) | 
                                                    (df_engineered['is_weekend'] == 1)).astype(int)
            
        return df_engineered

    def _filter_country_specific_columns(self, df, country_code):
        """
        Filter dataframe to keep only columns specific to the given country
        and general features that are not country-specific.
        
        Args:
            df: The input dataframe
            country_code: Either 'IT' or 'ES'
            
        Returns:
            Filtered dataframe
        """
        # Get all column names
        all_columns = df.columns.tolist()
        
        # Keep datetime and general features
        general_columns = ['DATETIME', 'day', 'hour', 'day_of_week', 'month', 
                          'quarter', 'is_weekend', 'hour_sin', 'hour_cos', 
                          'day_of_week_sin', 'day_of_week_cos', 'season', 
                          'is_business_hours', 'TMP_squared', 'heating_degree', 
                          'cooling_degree', 'SPV_log', 'hour_SPV', 'holiday_or_weekend']
        
        # Keep columns that contain the country code
        country_columns = [col for col in all_columns if country_code in col]
        
        # Also keep columns that contain VALUEMWHMETERINGDATA or INITIALROLLOUTVALUE for this country
        target_columns = [col for col in all_columns if f'VALUEMWHMETERINGDATA_customer{country_code}' in col or 
                          f'INITIALROLLOUTVALUE_customer{country_code}' in col]
        
        # Combine all columns to keep
        columns_to_keep = list(set(general_columns + country_columns + target_columns))
        columns_to_keep = [col for col in columns_to_keep if col in all_columns]
        
        return df[columns_to_keep]

    def __call__(self, train_data, input_data, target_columns):
        """
        Train models for IT and ES separately and predict target values.
        First trains on Italian data, then on Spanish data.
        
        Args:
            train_data: Training data
            input_data: Data to predict on
            target_columns: Target columns to predict
        
        Returns:
            DataFrame with predictions for target columns
        """
        # Initialize results dataframe
        df_result = pd.DataFrame(index=input_data.index, columns=target_columns)
        df_result = df_result.fillna(0.0).infer_objects(copy=False)
        
        # Create separate imputers for features and targets
        feature_imputer = SimpleImputer(strategy='mean')
        target_imputer = SimpleImputer(strategy='mean')
        
        # Process countries in order: IT first, then ES
        for country_code in ['IT', 'ES']:
            logger.info(f"Processing {country_code} data...")
            
            # Filter target columns for this country
            country_target_columns = [col for col in target_columns 
                                     if f'customer{country_code}' in col]
            
            if not country_target_columns:
                logger.info(f"No target columns found for {country_code}, skipping.")
                continue
                
            # Filter and engineer country-specific data
            train_data_country = self._filter_country_specific_columns(train_data, country_code)
            input_data_country = self._filter_country_specific_columns(input_data, country_code)
            
            # Add engineered features to both train and input data
            train_data_engineered = self._engineer_features(train_data_country, country_code)
            input_data_engineered = self._engineer_features(input_data_country, country_code)
            
            # Drop non-numeric columns, including timestamps, from engineered data
            train_data_numeric = train_data_engineered.select_dtypes(include=['number'])
            input_data_numeric = input_data_engineered.select_dtypes(include=['number'])

            # Process each target column for this country
            for target_column in tqdm(country_target_columns, 
                                    desc=f'Training XGBoost models for {country_code}'):
                try:
                    # Check if target column exists in train data
                    if target_column not in train_data_numeric.columns:
                        logger.warning(f"Target column {target_column} not found in training data. Skipping.")
                        df_result[target_column] = 0.0
                        continue
                    
                    # Extract the target column from train dataset
                    y = train_data_numeric[target_column].values
                    
                    # Check for NaN or empty values in target
                    if len(y) == 0 or np.isnan(y).all():
                        logger.warning(f"No valid data available for target column {target_column}. Skipping.")
                        df_result[target_column] = 0.0
                        continue
                    
                    # Remove NaN values from target and corresponding feature rows
                    valid_indices = ~np.isnan(y)
                    if np.sum(valid_indices) == 0:
                        logger.warning(f"No valid data after removing NaNs for {target_column}. Skipping.")
                        df_result[target_column] = 0.0
                        continue
                    
                    y = y[valid_indices]
                    
                    # Create feature matrix X by dropping the target columns from train_data
                    feature_columns = [col for col in train_data_numeric.columns 
                                     if col not in country_target_columns]
                    X_df = train_data_numeric[feature_columns]
                    
                    # Keep only rows with valid target values
                    X_df_valid = X_df.iloc[valid_indices]
                    
                    # Filter out columns with all NaN values or zero variance
                    non_nan_cols = X_df_valid.columns[~X_df_valid.isna().all()]
                    # Further filter to ensure we only keep columns with at least one non-missing value
                    valid_feature_cols = [col for col in non_nan_cols if not X_df_valid[col].isna().all()]
                    
                    # Skip if no valid features are available
                    if not valid_feature_cols:
                        logger.warning(f"No valid features available for target column {target_column}. Skipping.")
                        df_result[target_column] = 0.0
                        continue
                    
                    # Use only valid features and rows
                    X = X_df_valid[valid_feature_cols].values
                    
                    # Double-check dimensions before imputation
                    if X.shape[0] != y.shape[0] or X.shape[0] == 0:
                        logger.warning(f"Invalid dimensions after filtering: X shape {X.shape}, y shape {y.shape}. Skipping {target_column}.")
                        df_result[target_column] = 0.0
                        continue
                    
                    # Impute missing values using sklearn's SimpleImputer for X only
                    # We've already filtered y to remove NaNs
                    X = feature_imputer.fit_transform(X)

                    # Sample the data to reduce training time (if sample_size < 1)
                    if self.sample_size < 1.0 and len(X) > 1 and len(y) > 1:
                        # Calculate actual sample size - ensure at least 1 sample
                        n_samples = max(1, int(len(X) * self.sample_size))
                        
                        # Use simple random sampling
                        indices = np.random.RandomState(self.random_state).choice(
                            len(X), size=n_samples, replace=False
                        )
                        X_sampled = X[indices]
                        y_sampled = y[indices]
                    else:
                        X_sampled, y_sampled = X, y
                    
                    # Final verification of dimensions
                    if len(X_sampled) != len(y_sampled) or len(X_sampled) == 0 or len(y_sampled) == 0:
                        logger.warning(f"Dimension mismatch or empty arrays for {target_column}. X: {len(X_sampled)}, y: {len(y_sampled)}. Skipping.")
                        df_result[target_column] = 0.0
                        continue
                    
                    # Log the dimensions to help with debugging
                    logger.info(f"Training model for {target_column} with {X_sampled.shape[0]} samples and {X_sampled.shape[1]} features")
                        
                    # Fit the XGBoost model on the sampled data
                    self.model.fit(X_sampled, y_sampled)

                    # Prepare the input data for prediction (using the same valid features)
                    if not all(col in input_data_numeric.columns for col in valid_feature_cols):
                        missing_cols = [col for col in valid_feature_cols if col not in input_data_numeric.columns]
                        logger.warning(f"Missing feature columns in input data: {missing_cols}. This may affect prediction quality.")
                        # Only use feature columns that exist in input data
                        valid_feature_cols = [col for col in valid_feature_cols if col in input_data_numeric.columns]
                    
                    if not valid_feature_cols:
                        logger.warning(f"No valid features available in input data for {target_column}. Skipping.")
                        df_result[target_column] = 0.0
                        continue
                        
                    X_input = input_data_numeric[valid_feature_cols].values
                    
                    # Skip if input data has no valid features
                    if X_input.size == 0:
                        logger.warning(f"No valid input features for target column {target_column}. Skipping.")
                        df_result[target_column] = 0.0
                        continue
                    
                    # Make sure the input data has the right shape for imputation
                    if X_input.ndim == 1:
                        X_input = X_input.reshape(1, -1)
                        
                    X_input = feature_imputer.transform(X_input)

                    # Predict the target column for the input data
                    predictions = self.model.predict(X_input)
                    df_result[target_column] = predictions
                    
                except Exception as e:
                    logger.error(f"Error processing target column {target_column}: {str(e)}")
                    # In case of error, fill with zeros
                    df_result[target_column] = 0.0

        return df_result


