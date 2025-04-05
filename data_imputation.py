import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import heapq

csv_path_it = "/home/anna/Downloads/OneDrive_2025-04-05/Alpiq/datasets2025/historical_metering_data_IT.csv"
IT_dataframe = pd.read_csv(csv_path_it)

csv_path_sp = "/home/anna/Downloads/OneDrive_2025-04-05/Alpiq/datasets2025/historical_metering_data_SP.csv"
SP_dataframe = pd.read_csv(csv_path_sp)

csv_path_rollout_it = "/home/anna/Downloads/OneDrive_2025-04-05/Alpiq/datasets2025/rollout_data_IT.csv"
IT_rollout_dataframe = pd.read_csv(csv_path_it)

csv_path_rollout_sp = "/home/anna/Downloads/OneDrive_2025-04-05/Alpiq/datasets2025/rollout_data_ES.csv"
SP_rollout_dataframe = pd.read_csv(csv_path_sp)

IT_nan = IT_dataframe.isnull().sum()
SP_nan = SP_dataframe.isnull().sum()
IT_rollout_nan = IT_rollout_dataframe.isnull().sum()
SP_rollout_nan = SP_rollout_dataframe.isnull().sum()


IT_dense = IT_dataframe[IT_nan[IT_nan == 0].index]
SP_dense = SP_dataframe[SP_nan[SP_nan == 0].index]
IT_rollout_dense = IT_dataframe[IT_rollout_nan[IT_nan == 0].index]
SP_rollout_dense = SP_dataframe[SP_rollout_nan[SP_nan == 0].index]


IT_sparse = IT_dataframe[IT_nan[IT_nan > 0].index]
SP_sparse = SP_dataframe[SP_nan[SP_nan > 0].index]
IT_rollout_sparse = IT_dataframe[IT_rollout_nan[IT_nan > 0].index]
SP_rollout_sparse = SP_dataframe[SP_rollout_nan[SP_nan > 0].index]

def k_max_elements(lst, k):
    result = []
    for i, value in enumerate(lst):
        if i < k:
            heapq.heappush(result, (value, i))
        else:
            heapq.heappushpop(result, (value, i))
    return sorted(result, reverse=True)

def mean_imputation(df_dense, df_sparse):
    dense_array = df_dense.to_numpy()
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(dense_array)
    transformed_array = imp.transform(df_sparse)
    df_sparse = pd.DataFrame(transformed_array)
    df_dense = pd.DataFrame(dense_array)
    imputed_result = df_dense.merge(df_sparse)
    return imputed_result



def hot_deck_imputation(df_dense, df_sparse, k):
    dense_array = df_dense.to_numpy()
    sparse_array = df_sparse.to_numpy()
    for i in range(len(sparse_array[0])):
        column_toimpute = sparse_array[:, i]
        similarity_array = []
        empty_entries = np.isnan(column_toimpute)
        for j in range(len(dense_array[0])):
            measure_column = dense_array[:, j][~empty_entries]
            column_info = column_toimpute[~empty_entries]
            similarity = np.dot(measure_column, column_info)
            similarity_array.append(similarity)
            top_k_list = k_max_elements(similarity)
            top_k_indices = [j for (i,j) in top_k_list]
            column_toimpute[empty_entries] = dense_array[empty_entries, top_k_indices].mean()
    df_sparse = pd.DataFrame(df_sparse)
    df_dense = pd.DataFrame(df_dense)
    imputed_result = df_dense.merge(df_sparse)
    return imputed_result

def knn_imputation(df_dense, df_sparse):
    dense_array = df_dense.to_numpy()
    imp = KNNImputer(n_neighbors=2, weights="uniform")
    imp.fit(dense_array)
    transformed_array = imp.transform(df_sparse)
    df_sparse = pd.DataFrame(transformed_array)
    df_dense = pd.DataFrame(dense_array)
    imputed_result = df_dense.merge(df_sparse)
    return imputed_result


def prediction_vs_reality (impute_fct, predicted_df, reality_df):
    predicted_array = predicted_df.to_numpy()
    reality_array = reality_df.to_numpy()
    empty_entries = np.isnan(predicted_array)
    difference_array = np.zeros((len(predicted_array), len(predicted_array[0])))
    for i in range(len(predicted_array[0])):
        predicted_info = predicted_array[:, i].isnan()
        reality_info = reality_array[:, i].isnan()
        difference_info = [predicted_info[j] and reality_info[j] for j in range(len(reality_info))]
        difference_array[:, i] = [reality_info[j] - predicted_info[j] if difference_info[j] == True else np.nan for j in range(len(reality_info))]
    
    difference_imputed = impute_fct(difference_array)
    df_difference = pd.DataFrame(difference_imputed)
    return df_difference
            


