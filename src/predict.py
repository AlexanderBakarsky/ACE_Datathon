from evaluator import Evaluator
from dataloader import Dataset
from models.medium_xgboost import XGBoostModel
import pandas as pd
import numpy as np
import logging
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_prediction_month(dataset_path):
    """
    Find the month where VALUEMWHMETERINGDATA columns stop but other data continues
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        start_date: Start date of the prediction period
        end_date: End date of the prediction period
    """
    # Load the dataset
    dataset = Dataset(dataset_path)
    
    # Get the full merged dataframe
    full_data = dataset.data_merged.copy()
    
    # Get all VALUEMWHMETERINGDATA columns for both countries
    metering_columns_it = [col for col in full_data.columns if 'VALUEMWHMETERINGDATA_customerIT' in col]
    metering_columns_es = [col for col in full_data.columns if 'VALUEMWHMETERINGDATA_customerES' in col]
    metering_columns = metering_columns_it + metering_columns_es
    
    # Find the last date with valid metering data
    last_valid_dates = []
    for col in metering_columns:
        # Find the last non-NaN entry in each column
        last_valid = full_data[col].last_valid_index()
        if last_valid is not None:
            last_valid_dates.append(last_valid)
    
    if not last_valid_dates:
        raise ValueError("No valid metering data found")
    
    # Get the latest date with valid metering data
    last_metering_date = max(last_valid_dates)
    logger.info(f"Last date with metering data: {last_metering_date}")
    
    # Make sure last_metering_date is a datetime object
    if not isinstance(last_metering_date, pd.Timestamp):
        # Convert to datetime if it's a string
        if isinstance(last_metering_date, str):
            last_metering_date = pd.to_datetime(last_metering_date)
        # If it's an integer (like 22628), assume it's a row index and get the actual date from the DATETIME column
        elif isinstance(last_metering_date, (int, np.integer)):
            last_metering_date = full_data.loc[last_metering_date, 'DATETIME']
            logger.info(f"Converted index to datetime: {last_metering_date}")
    
    # Check if there is data after this date
    data_after = full_data[full_data['DATETIME'] > last_metering_date]
    
    if len(data_after) == 0:
        raise ValueError("No data available after the last metering date")
    
    # Find the end date of the available data
    end_date = data_after['DATETIME'].max()
    
    # Return start_date (one day after last metering) and end_date
    start_date = last_metering_date + pd.Timedelta(days=1)
    
    logger.info(f"Prediction period: {start_date} to {end_date}")
    return start_date, end_date

def prepare_prediction_data(dataset_path):
    """
    Prepare the data for prediction, using all available data up to the prediction month for training
    
    Args:
        dataset_path: Path to the dataset
        
    Returns:
        train_data: Training data
        input_data: Input data for prediction
        target_columns: Target columns to predict
    """
    # Load the dataset
    dataset = Dataset(dataset_path)
    
    # Get start and end dates for prediction
    start_date, end_date = find_prediction_month(dataset_path)
    
    # Create train and input data splits
    full_data = dataset.data_merged.copy()
    train_data = full_data[full_data['DATETIME'] < start_date]
    input_data = full_data[(full_data['DATETIME'] >= start_date) & (full_data['DATETIME'] <= end_date)]
    
    # Get target columns
    metering_columns_it = [col for col in full_data.columns if 'VALUEMWHMETERINGDATA_customerIT' in col]
    metering_columns_es = [col for col in full_data.columns if 'VALUEMWHMETERINGDATA_customerES' in col]
    target_columns = metering_columns_it + metering_columns_es
    
    # Use the same column order as in the evaluator
    target_columns.sort()
    
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Input data shape: {input_data.shape}")
    logger.info(f"Number of target columns: {len(target_columns)}")
    
    return train_data, input_data, target_columns

def save_predictions(output, output_path, team_name="Team"):
    """
    Save the prediction results to CSV files, one for each country
    
    Args:
        output: DataFrame with predictions
        output_path: Path to save the output
        team_name: Name of the team for the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Split the output by country
    italy_columns = [col for col in output.columns if 'customerIT' in col]
    spain_columns = [col for col in output.columns if 'customerES' in col]
    
    # Save the results
    if italy_columns:
        italy_output = output[italy_columns]
        italy_file = os.path.join(output_path, f"students_results_{team_name}_IT.csv")
        italy_output.to_csv(italy_file)
        logger.info(f"Saved Italian predictions to {italy_file}")
    
    if spain_columns:
        spain_output = output[spain_columns]
        spain_file = os.path.join(output_path, f"students_results_{team_name}_ES.csv")
        spain_output.to_csv(spain_file)
        logger.info(f"Saved Spanish predictions to {spain_file}")
    
    # Save combined output
    combined_file = os.path.join(output_path, f"students_results_{team_name}_combined.csv")
    output.to_csv(combined_file)
    logger.info(f"Saved combined predictions to {combined_file}")

def main():
    # Paths
    dataset_path = '~/Documents/Projects/ACE_Datathon/datasets2025'
    output_path = '~/Documents/Projects/ACE_Datathon/outputs'
    team_name = "Totoro"
    
    # Process the data
    logger.info("Preparing data for prediction...")
    train_data, input_data, target_columns = prepare_prediction_data(dataset_path)
    
    # Initialize the model
    logger.info("Initializing model...")
    model = XGBoostModel()
    
    # Generate predictions
    logger.info("Generating predictions...")
    output = model(train_data, input_data, target_columns)
    
    # Save the results
    logger.info("Saving predictions...")
    save_predictions(output, output_path, team_name)
    
    logger.info("Prediction completed successfully!")
    return output

if __name__ == "__main__":
    main()