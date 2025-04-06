from os.path import join
import pandas as pd

class Dataset:
    def __init__(self,
                 data_path: str,
                 ):
        
        spv_data_path = join(data_path, 'spv_ec00_forecasts_es_it.xlsx')
        og_metering_data_path_spain = join(data_path, 'historical_metering_data_ES.csv')
        og_metering_data_path_italy = join(data_path, 'historical_metering_data_IT.csv')
        og_rollout_data_path_spain = join(data_path, 'rollout_data_ES.csv')
        og_rollout_data_path_italy = join(data_path, 'rollout_data_IT.csv')


        metering_data_path_spain = join(data_path, 'imputed_ES_usage.csv')
        metering_data_path_italy = join(data_path, 'imputed_IT_usage.csv')
        rollout_data_path_spain = join(data_path, 'imputed_ES_rollout.csv')
        rollout_data_path_italy = join(data_path, 'imputed_IT_rollout.csv') 

        holiday_data_path_spain = join(data_path, 'holiday_ES.xlsx')
        holiday_data_path_italy = join(data_path, 'holiday_IT.xlsx')
        
        # set the name of the columns of the spv data, note that the spv data is for ES and IT in different sheets
        self.spv_data_es = pd.read_excel(spv_data_path, sheet_name='ES',
                                      names=['DATETIME', 'SPV_ES', 'TMP_ES'])
        self.spv_data_it = pd.read_excel(spv_data_path, sheet_name='IT',
                                      names=['DATETIME', 'SPV_IT', 'TMP_IT'])
        # Combine ES and IT data
        self.metering_data_spain = pd.read_csv(metering_data_path_spain)
        self.metering_data_italy = pd.read_csv(metering_data_path_italy)
        self.rollout_data_spain = pd.read_csv(rollout_data_path_spain)
        self.rollout_data_italy = pd.read_csv(rollout_data_path_italy)
        self.holiday_data_spain = pd.read_excel(holiday_data_path_spain)
        self.holiday_data_italy = pd.read_excel(holiday_data_path_italy)

        # Add the DATETIME column from og metering data to the imputed data
        self.metering_data_spain['DATETIME'] = pd.read_csv(og_metering_data_path_spain)['DATETIME']
        self.metering_data_italy['DATETIME'] = pd.read_csv(og_metering_data_path_italy)['DATETIME']
        self.rollout_data_spain['DATETIME'] = pd.read_csv(og_rollout_data_path_spain)['DATETIME']
        self.rollout_data_italy['DATETIME'] = pd.read_csv(og_rollout_data_path_italy)['DATETIME']
        
        # convert the datetime to datetime
        self.spv_data_es['DATETIME'] = pd.to_datetime(self.spv_data_es['DATETIME'])
        self.spv_data_it['DATETIME'] = pd.to_datetime(self.spv_data_it['DATETIME'])
        self.metering_data_spain['DATETIME'] = pd.to_datetime(self.metering_data_spain['DATETIME'])
        self.metering_data_italy['DATETIME'] = pd.to_datetime(self.metering_data_italy['DATETIME'])
        self.rollout_data_spain['DATETIME'] = pd.to_datetime(self.rollout_data_spain['DATETIME'])
        self.rollout_data_italy['DATETIME'] = pd.to_datetime(self.rollout_data_italy['DATETIME'])
        
        
        self.spv_data_es_start = self.spv_data_es['DATETIME'].min()
        self.spv_data_es_end = self.spv_data_es['DATETIME'].max()
        self.spv_data_it_start = self.spv_data_it['DATETIME'].min()
        self.spv_data_it_end = self.spv_data_it['DATETIME'].max()
        
        self.metering_data_spain_start = self.metering_data_spain['DATETIME'].min()
        self.metering_data_spain_end = self.metering_data_spain['DATETIME'].max()
        self.metering_data_italy_start = self.metering_data_italy['DATETIME'].min()
        self.metering_data_italy_end = self.metering_data_italy['DATETIME'].max()
        
        self.rollout_data_spain_start = self.rollout_data_spain['DATETIME'].min()
        self.rollout_data_spain_end = self.rollout_data_spain['DATETIME'].max()
        self.rollout_data_italy_start = self.rollout_data_italy['DATETIME'].min()
        self.rollout_data_italy_end = self.rollout_data_italy['DATETIME'].max()
        
        self.holiday_data_spain['holiday_ES'] = pd.to_datetime(self.holiday_data_spain['holiday_ES']).dt.date
        self.holiday_data_italy['holiday_IT'] = pd.to_datetime(self.holiday_data_italy['holiday_IT']).dt.date
        
        self.holiday_data_spain_start = self.holiday_data_spain['holiday_ES'].min()
        self.holiday_data_spain_end = self.holiday_data_spain['holiday_ES'].max()
        self.holiday_data_italy_start = self.holiday_data_italy['holiday_IT'].min()
        self.holiday_data_italy_end = self.holiday_data_italy['holiday_IT'].max()
        
        # Clip SPV dataset based on the max of rollout data
        # Find the max end date from rollout data
        max_rollout_end = max(self.rollout_data_spain_end, self.rollout_data_italy_end)
        
        # Clip SPV data to only include dates up to the max rollout end date
        self.spv_data_es = self.spv_data_es[self.spv_data_es['DATETIME'] <= max_rollout_end]
        self.spv_data_it = self.spv_data_it[self.spv_data_it['DATETIME'] <= max_rollout_end]
        
        # Update the SPV data end dates after clipping
        self.spv_data_es_end = self.spv_data_es['DATETIME'].max()
        self.spv_data_it_end = self.spv_data_it['DATETIME'].max()
        
        # merge SPV, metering and rollout data for each country, use NULL for missing values
        self.data_spain = pd.merge(self.rollout_data_spain, self.metering_data_spain, on='DATETIME', how='outer')
        self.data_italy = pd.merge(self.rollout_data_italy, self.metering_data_italy, on='DATETIME', how='outer')
        
        # merge spv
        self.data_spain = pd.merge(self.data_spain, self.spv_data_es, on='DATETIME', how='outer')
        self.data_italy = pd.merge(self.data_italy, self.spv_data_it, on='DATETIME', how='outer')
      
        # merge the two countries dataframes
        self.data_merged = pd.merge(self.data_spain, self.data_italy, on='DATETIME', how='outer')
        
        # merge the data_merged with the holiday data, note that the holiday data is for days but the data_merged is for hours
        self.data_merged['day'] = self.data_merged['DATETIME'].dt.date
        
        # add a column for each row to indicate if it is a holiday or not
        self.data_merged['is_holiday_ES'] = self.data_merged['day'].isin(self.holiday_data_spain['holiday_ES'])
        self.data_merged['is_holiday_IT'] = self.data_merged['day'].isin(self.holiday_data_italy['holiday_IT'])
        
        # validation split intervals (A, B, C) -> train set [A, B), test [B, C)
        
        self.validation_split_intervals = [
            ('2022-01-01 00:00:00', '2023-08-01 00:00:00', '2023-09-01 00:00:00'),
            ('2022-01-01 00:00:00', '2023-10-01 00:00:00', '2023-11-01 00:00:00'),
            ('2022-01-01 00:00:00', '2023-12-01 00:00:00', '2024-01-01 00:00:00'),
            ('2022-01-01 00:00:00', '2024-02-01 00:00:00', '2024-03-01 00:00:00'),
            ('2022-01-01 00:00:00', '2024-04-01 00:00:00', '2024-05-01 00:00:00'),
            ('2022-01-01 00:00:00', '2024-07-01 00:00:00', '2024-08-01 00:00:00'),
        ]
    
    def get_train_test_split(self, split_index=0):
        """
        Get train and test data based on validation split intervals.
        
        Args:
            split_index (int): Index of the validation split interval to use.
                               Defaults to 0.
        
        Returns:
            tuple: (train_data, test_data) pandas DataFrames
        """
        if split_index < 0 or split_index >= len(self.validation_split_intervals):
            raise ValueError(f"Split index must be between 0 and {len(self.validation_split_intervals) - 1}")
        
        # Get the split interval
        train_start, train_end, test_end = self.validation_split_intervals[split_index]
        
        # Convert string dates to datetime
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        test_end = pd.to_datetime(test_end)
        
        # Create train and test datasets
        train_data = self.data_merged[(self.data_merged['DATETIME'] >= train_start) & 
                                      (self.data_merged['DATETIME'] < train_end)]
        
        test_data = self.data_merged[(self.data_merged['DATETIME'] >= train_end) & 
                                     (self.data_merged['DATETIME'] < test_end)]
        
        # split the test data into target value and input values
        # target columns are ones with 'VALUEMWHMETERINGDATA' in the column name
        target_columns = [col for col in test_data.columns if 'VALUEMWHMETERINGDATA' in col]
        input_columns = [col for col in test_data.columns if col not in target_columns]
        
        
        target_columns = ["DATETIME"] + target_columns
        # split the test data into target and input
        target_data = test_data[target_columns]
        input_data = test_data[input_columns]
        
        return train_data, input_data, target_data
    
    def __getitem__(self, index):
        """
        Get a specific validation split.
        
        Args:
            index (int): Index of the validation split to retrieve.
        
        Returns:
            tuple: (train_data, test_data) for the specified split
        """
        return self.get_train_test_split(index)
    
    def __len__(self):
        """
        Return the number of validation splits available.
        
        Returns:
            int: Number of validation splits
        """
        return len(self.validation_split_intervals)
    
    
        
    
