import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import heapq

# csv_path_it = "/home/anna/Hackaton_Project/ace_datathon/datasets2025/historical_metering_data_IT.csv"
# IT_dataframe = pd.read_csv(csv_path_it).drop("DATETIME", axis=1)

# csv_path_sp = "/home/anna/Hackaton_Project/ace_datathon/datasets2025/historical_metering_data_ES.csv"
# SP_dataframe = pd.read_csv(csv_path_sp).drop("DATETIME", axis=1)

# csv_path_rollout_it = "/home/anna/Hackaton_Project/ace_datathon/datasets2025/rollout_data_IT.csv"
# IT_rollout_dataframe = pd.read_csv(csv_path_it).drop("DATETIME", axis=1)

# csv_path_rollout_sp = "/home/anna/Hackaton_Project/ace_datathon/datasets2025/rollout_data_ES.csv"
# SP_rollout_dataframe = pd.read_csv(csv_path_sp).drop("DATETIME", axis=1)

# IT_nan = IT_dataframe.isna().sum()
# SP_nan = SP_dataframe.isna().sum()
# IT_rollout_nan = IT_rollout_dataframe.isna().sum()
# SP_rollout_nan = SP_rollout_dataframe.isna().sum()


# IT_dense = IT_dataframe[IT_nan[IT_nan == 0].index]
# SP_dense = SP_dataframe[SP_nan[SP_nan == 0].index]
# IT_rollout_dense = IT_dataframe[IT_rollout_nan[IT_nan == 0].index]
# SP_rollout_dense = SP_dataframe[SP_rollout_nan[SP_nan == 0].index]


# IT_sparse = IT_dataframe[IT_nan[IT_nan > 0].index]
# SP_sparse = SP_dataframe[SP_nan[SP_nan > 0].index]
# IT_rollout_sparse = IT_dataframe[IT_rollout_nan[IT_nan > 0].index]
# SP_rollout_sparse = SP_dataframe[SP_rollout_nan[SP_nan > 0].index]

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

def normalize(v):
    return v / np.linalg.norm(v)

def hot_deck_imputation(df_dense, df_sparse, k=1):
    dense_array = df_dense.to_numpy()
    sparse_cols = df_sparse.columns
    sparse_array = df_sparse.to_numpy()
    for i in range(len(sparse_array[0])):
        print(f"Imputing sparse column ({i}/{len(sparse_array[0])})")
        impute_client = sparse_array[:, i]
        similarity_array = []
        # Get all empty predictions for a specific client
        empty_entries = np.isnan(impute_client)
        for j in range(len(dense_array[0])):
            # For the previously selected client
            # We need to make doc product possible => 
            # comparing only dimensions available on both
            gt_non_empty_dim = dense_array[:, j][~empty_entries]
            client_non_empty_dim = impute_client[~empty_entries]
            # TODO: Try converting to unit vectors
            similarity = np.dot(normalize(gt_non_empty_dim), normalize(client_non_empty_dim))
            similarity_array.append(similarity)
        top_k_list = k_max_elements(similarity_array, k)
        print(top_k_list)
        top_k_indices = [index for (value, index) in top_k_list]
        # Set all empty imputee fields to the closest dense neighbour's values
        # Empty imputee fields = empty_entries
        # Closest dense neighbour = top_k
        ground_truth_dense = np.mean(dense_array[:,top_k_indices], axis=1)
        print(ground_truth_dense)
        impute_client[empty_entries] = ground_truth_dense[empty_entries]

    df_sparse = pd.DataFrame(sparse_array, columns=sparse_cols)
    print("Imputation success?")
    print(sum(df_sparse.isna().sum()) == 0)
    return df_sparse

def knn_imputation(df_dense, df_sparse):
    dense_array = df_dense.to_numpy()
    print(dense_array)
    sparse_array = df_sparse.to_numpy()

    print("Sparse array:")
    print(sparse_array.shape)

    print("Dense array:")
    print(dense_array.shape)
    imp = KNNImputer(n_neighbors=2, weights="uniform")
    imp.fit(dense_array)
    transformed_array = imp.transform(sparse_array)
    df_sparse = pd.DataFrame(transformed_array)
    df_dense = pd.DataFrame(dense_array)
    imputed_result = pd.concat([df_dense, df_sparse], axis=1)
    return imputed_result


def prediction_vs_reality (impute_fct, predicted_df, reality_df):
    predicted_array = predicted_df.to_numpy()
    reality_array = reality_df.to_numpy()
    empty_entries = pd.isnull(predicted_df)
    empty_entries = empty_entries.to_numpy()
    difference_array = np.zeros((len(predicted_array), len(predicted_array[0])))

    print("Computing differences...")
    for i in range(len(predicted_array[0])):
        predicted_info = pd.DataFrame(predicted_array[:, i]).isnull().to_numpy()
        reality_info = pd.DataFrame(reality_array[:, i]).isnull().to_numpy()
        
        difference_info = [predicted_info[j] and reality_info[j] for j in range(len(reality_info))]
        #print(reality_array)
        
        for j in range(len(reality_info)):
            if difference_info[j] == True:
                difference_array[j, i] = reality_array[j,i] - predicted_array[j,i]
            else:
                difference_array[j, i] = np.nan
    
    difference = pd.DataFrame(difference_array)
    diff_nan = difference.isnull().sum()
    difference_sparse = difference[diff_nan[diff_nan > 0].index]
    difference_dense = difference[diff_nan[diff_nan == 0].index]

    print("Calling impute fn:")
    print(impute_fct.__name__)
    difference_imputed = impute_fct(difference_dense, difference_sparse)

    return difference_imputed
            


