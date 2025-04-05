from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm


class SimpleRegression:
    def __init__(self,
                 ):
        self.model = LinearRegression(n_jobs=-1)
        self.window_size = 48
        
    def __call__(self,
                 train_data,
                 input_data,
                 target_columns,
                 ):
        
        # train data is a dataframe of features and target columns
        # input data is a dataframe of features
        # target columns is a list of target columns

        window_size = self.window_size
        df_result = pd.DataFrame(index=input_data.index, columns=target_columns)
        # set the default value to 0 - using the recommended approach to avoid the warning
        df_result = df_result.fillna(0.0).infer_objects(copy=False)
  
        for target_column in tqdm(target_columns, desc='Training models'):
            # extract the column from train dataset
            data = train_data[target_column]
            data = np.array(data)
            
            # impute the missing values with the mean of the column
            # Check if there are any non-NaN values before calculating mean
            if np.isnan(data).all():
                # Skip this column if all values are NaN
                continue
                
            # Calculate mean only on non-NaN values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                data = np.where(np.isnan(data), np.nanmean(valid_data), data)
            else:
                # Skip this column if there are no valid values
                continue

                
            # data is a sequence of values of shape [n_samples], we need to build a numpy matrix with shape [n_samples - window size + 1, window size]
            X = np.array([data[i:i+window_size] for i in range(len(data) - window_size)])
            y = np.array([data[i+window_size] for i in range(len(data) - window_size)])
            
            random_indexes = np.random.permutation(len(X))[:100]
            X = X[random_indexes]
            y = y[random_indexes]
   
            # fit the model on training data
            self.model.fit(X, y)
   
            # predict the target column
            last_window = data[-window_size:]
            last_window = np.array(last_window).reshape(1, -1)
            for i in range(len(input_data)):
                last_window = np.array(last_window).reshape(1, -1)
                prediction = self.model.predict(last_window)
                df_result.loc[input_data.index[i], target_column] = prediction[0]
                last_window = np.append(last_window[0][1:], prediction[0])
            
        return df_result
        
        
        