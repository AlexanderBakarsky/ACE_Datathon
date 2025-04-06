import data_imputation
from statsmodels.tsa.arima.model import ARIMA, sarimax
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd
import numpy as np

#### ITALY DATA
csv_path_it = "/home/anna/Hackaton_Project/ace_datathon/datasets2025/historical_metering_data_IT.csv"
IT_dataframe = pd.read_csv(csv_path_it).drop("DATETIME", axis=1)
#IT_dataframe = IT_dataframe[(IT_dataframe.isnull().all(axis=1)==False).index]
#print(IT_dataframe)
#print(sum(IT_dataframe.isna().all().to_list()))
csv_path_rollout_it = "/home/anna/Hackaton_Project/ace_datathon/datasets2025/rollout_data_IT.csv"
IT_rollout_dataframe = pd.read_csv(csv_path_rollout_it).drop("DATETIME", axis=1)

IT_nan = IT_dataframe.isna().sum()
IT_rollout_nan = IT_rollout_dataframe.isna().sum()

IT_dense = IT_dataframe[IT_nan[IT_nan == 0].index]
IT_sparse = IT_dataframe[IT_nan[IT_nan > 0].index]

IT_rollout_dense = IT_rollout_dataframe[IT_rollout_nan[IT_rollout_nan == 0].index]
IT_rollout_sparse = IT_rollout_dataframe[IT_rollout_nan[IT_rollout_nan > 0].index]

#### SPAIN DATA
csv_path_es = "/home/anna/Hackaton_Project/ace_datathon/datasets2025/historical_metering_data_ES.csv"
ES_dataframe = pd.read_csv(csv_path_es).drop("DATETIME", axis=1)
#IT_dataframe = IT_dataframe[(IT_dataframe.isnull().all(axis=1)==False).index]
#print(IT_dataframe)
#print(sum(IT_dataframe.isna().all().to_list()))
csv_path_rollout_es = "/home/anna/Hackaton_Project/ace_datathon/datasets2025/rollout_data_ES.csv"
ES_rollout_dataframe = pd.read_csv(csv_path_rollout_es).drop("DATETIME", axis=1)

ES_nan = ES_dataframe.isna().sum()
ES_rollout_nan = ES_rollout_dataframe.isna().sum()

ES_dense = ES_dataframe[ES_nan[ES_nan == 0].index]
ES_sparse = ES_dataframe[ES_nan[ES_nan > 0].index]

ES_rollout_dense = ES_rollout_dataframe[ES_rollout_nan[ES_rollout_nan == 0].index]
ES_rollout_sparse = ES_rollout_dataframe[ES_rollout_nan[ES_rollout_nan > 0].index]

def impute(dense, sparse, country, type):
    df_sparse_imputed = data_imputation.hot_deck_imputation(dense, sparse, k=3)
    imputed_result = pd.concat([dense, df_sparse_imputed], axis=1)    
    imputed_result.to_csv(f"imputed_{country}_{type}.csv")
    return imputed_result

if __name__ == "__main__":
    print("Imputing usage...")
    impute(IT_dense, IT_sparse, "IT", "usage")
    
    print("Imputing rollout...")
    impute(IT_rollout_dense, IT_rollout_sparse, "IT", "rollout")
    
    print("Imputing usage...")
    impute(ES_dense, ES_sparse, "ES", "usage")
    
    print("Imputing rollout...")
    impute(ES_rollout_dense, ES_rollout_sparse, "ES", "rollout")
    
    '''imp_csv_path_es = "/home/anna/Hackaton_Project/ace_datathon/imputed_ES_usage.csv"
    imp_csv_path_rollout_es = "/home/anna/Hackaton_Project/ace_datathon/imputed_ES_rollout.csv"
    imp_csv_path_it = "/home/anna/Hackaton_Project/ace_datathon/imputed_IT_usage.csv"
    imp_csv_path_rollout_it = "/home/anna/Hackaton_Project/ace_datathon/imputed_IT_rollout.csv"

    imp_IT_dataframe = pd.read_csv(imp_csv_path_it)
    imp_IT_rollout_dataframe = pd.read_csv(imp_csv_path_rollout_it)
    imp_IT_dataframe = imp_IT_dataframe.reindex(sorted(imp_IT_dataframe.columns), axis=1)
    imp_IT_rollout_dataframe = imp_IT_rollout_dataframe.reindex(sorted(imp_IT_rollout_dataframe.columns), axis=1)
    #print(imp_IT_dataframe.shape)
    #print(imp_IT_rollout_dataframe.shape)
    imp_IT_array = imp_IT_dataframe.to_numpy()
    imp_IT_rollout_array = imp_IT_rollout_dataframe.to_numpy()
    to_cut = len(imp_IT_array)
    imp_IT_rollout_array = imp_IT_rollout_array[:to_cut, :]
    print(imp_IT_rollout_array.shape)
    
    diff_dataframe = imp_IT_array - imp_IT_rollout_array
    print(diff_dataframe.shape)
    diff_array = diff_dataframe
    diff_len = len(diff_array)
    diff_array_train = diff_array[:int(5*diff_len/6), :]
    diff_array_test = diff_array[int(5*diff_len/6):, :]
    print(diff_array_train.shape)
    print(diff_array_test.shape)
    diff_test = diff_len - int(5*diff_len/6)
    test_rollout = imp_IT_rollout_array[int(5*diff_len/6):, :]
    
    print(test_rollout.shape)
    
    arima_result = np.zeros((diff_test,  len(diff_array[0])))
    print(arima_result.shape)
    for i in range(len(diff_array[0])):
        train_i = diff_array_train[i]
        model = sarimax.SARIMAX(train_i, order=(1,1,1))
        model_fit = model.fit()
        diff_forecast = model_fit.forecast(steps = diff_test)
        arima_result[:, i] = diff_forecast
    
    loss = abs(diff_array_test - (arima_result + test_rollout)).sum()
    
    print(loss)'''
        
