import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta, datetime
import matplotlib.dates as mdates
import logging
from dataloader import Dataset
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default number of days of historical data to show
DEFAULT_LOOKBACK_DAYS = 30  # Show 1 month of historical data

def load_predictions(predictions_path, team_name="Team"):
    """
    Load prediction results from CSV files
    
    Args:
        predictions_path: Path to the prediction files
        team_name: Name of the team for the output file
        
    Returns:
        predictions_df: DataFrame with predictions
    """
    # Construct file paths
    combined_file = os.path.join(predictions_path, f"students_results_{team_name}_combined.csv")
    
    # Check if the combined file exists
    if os.path.exists(combined_file):
        logger.info(f"Loading combined predictions from {combined_file}")
        predictions_df = pd.read_csv(combined_file)
        
        # Handle datetime index
        if 'DATETIME' in predictions_df.columns:
            predictions_df['DATETIME'] = pd.to_datetime(predictions_df['DATETIME'])
            # Set DATETIME as index
            predictions_df.set_index('DATETIME', inplace=True)
        elif 'Unnamed: 0' in predictions_df.columns:
            # Sometimes the index gets saved as 'Unnamed: 0'
            # Ensure it's properly parsed as datetime
            try:
                predictions_df['DATETIME'] = pd.to_datetime(predictions_df['Unnamed: 0'])
                predictions_df.set_index('DATETIME', inplace=True)
                predictions_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
            except:
                # If it can't be parsed as datetime, create a datetime index for 2024
                logger.warning("Could not parse date column, creating synthetic index for 2024")
                start_date = datetime(2024, 1, 1)
                num_rows = predictions_df.shape[0]
                date_range = pd.date_range(start=start_date, periods=num_rows, freq='D')
                predictions_df.index = date_range
                predictions_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
        else:
            # No datetime column, create a synthetic index for 2024
            logger.warning("No datetime column found, creating synthetic index for 2024")
            start_date = datetime(2024, 1, 1) 
            num_rows = predictions_df.shape[0]
            date_range = pd.date_range(start=start_date, periods=num_rows, freq='D')
            predictions_df.index = date_range
        
        return predictions_df
    else:
        # Try to load individual country files
        italy_file = os.path.join(predictions_path, f"students_results_{team_name}_IT.csv")
        spain_file = os.path.join(predictions_path, f"students_results_{team_name}_ES.csv")
        
        dfs = []
        
        if os.path.exists(italy_file):
            logger.info(f"Loading Italian predictions from {italy_file}")
            italy_df = pd.read_csv(italy_file)
            dfs.append(italy_df)
            
        if os.path.exists(spain_file):
            logger.info(f"Loading Spanish predictions from {spain_file}")
            spain_df = pd.read_csv(spain_file)
            dfs.append(spain_df)
            
        if not dfs:
            raise FileNotFoundError(f"No prediction files found for team {team_name}")
            
        # Combine the dataframes
        predictions_df = pd.concat(dfs, axis=1)
        
        # Handle datetime index
        if 'DATETIME' in predictions_df.columns:
            predictions_df['DATETIME'] = pd.to_datetime(predictions_df['DATETIME'])
            # Set DATETIME as index
            predictions_df.set_index('DATETIME', inplace=True)
        elif 'Unnamed: 0' in predictions_df.columns:
            # Try to parse as datetime
            try:
                predictions_df['DATETIME'] = pd.to_datetime(predictions_df['Unnamed: 0'])
                predictions_df.set_index('DATETIME', inplace=True)
                predictions_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
            except:
                # If can't parse, create a synthetic index for 2024
                logger.warning("Could not parse date column, creating synthetic index for 2024")
                start_date = datetime(2024, 1, 1)
                num_rows = predictions_df.shape[0]
                date_range = pd.date_range(start=start_date, periods=num_rows, freq='D')
                predictions_df.index = date_range
                predictions_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
        else:
            # No datetime column, create a synthetic index for 2024
            logger.warning("No datetime column found, creating synthetic index for 2024")
            start_date = datetime(2024, 1, 1)
            num_rows = predictions_df.shape[0]
            date_range = pd.date_range(start=start_date, periods=num_rows, freq='D')
            predictions_df.index = date_range
            
        return predictions_df

def plot_predictions_with_history(dataset_path, predictions_path, output_path, team_name="Team", lookback_days=DEFAULT_LOOKBACK_DAYS):
    """
    Plot the predicted data alongside the historical data
    
    Args:
        dataset_path: Path to the dataset
        predictions_path: Path to the prediction files 
        output_path: Path to save the plots
        team_name: Name of the team for the output file
        lookback_days: Number of days of historical data to show before predictions (default: 30 days/1 month)
    """
    # Load the dataset
    dataset = Dataset(dataset_path)
    
    # Load predictions
    predictions_df = load_predictions(predictions_path, team_name)
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Get all customer columns from predictions
    italy_columns = [col for col in predictions_df.columns if 'customerIT' in col]
    spain_columns = [col for col in predictions_df.columns if 'customerES' in col]   
    
    # Get historical data up to 2024-07-31
    historical_end_date = pd.Timestamp('2024-07-31 23:59:59')
    
    # Process historical data
    historical_data = dataset.data_merged.copy()
    historical_data['DATETIME'] = pd.to_datetime(historical_data['DATETIME'])
    historical_data = historical_data.set_index('DATETIME')
    
    # Filter historical data up to historical_end_date
    historical_data = historical_data[historical_data.index <= historical_end_date]
    
    if historical_data.empty:
        logger.warning("No historical data found up to 2024-07-31, please check your dataset")
        return
    
    # Calculate the date 5 months before the end of historical data
    five_months_before = historical_end_date - pd.DateOffset(months=5)
    
    # Filter to show only the last 5 months of historical data
    historical_data = historical_data[historical_data.index >= five_months_before]
    
    logger.info(f"Filtered historical data to last 5 months: {historical_data.index.min()} to {historical_data.index.max()}")
    
    # Create a new date range for predictions starting from 2024-08-01
    prediction_start_date = pd.Timestamp('2024-08-01 00:00:00')
    prediction_end_date = pd.Timestamp('2024-08-31 23:00:00')  # End of August 2024
    
    # Create hourly time points for August 2024
    prediction_dates = pd.date_range(start=prediction_start_date, 
                                    end=prediction_end_date, 
                                    freq='H')
    
    logger.info(f"Creating prediction range from {prediction_start_date} to {prediction_end_date} with {len(prediction_dates)} hourly points")
    
    # Number of prediction hours (24 hours * 31 days = 744 hours for August)
    num_prediction_hours = len(prediction_dates)
    
    # If predictions_df has fewer rows than needed, repeat or truncate as necessary
    if len(predictions_df) < num_prediction_hours:
        logger.warning(f"Prediction data has fewer rows ({len(predictions_df)}) than needed ({num_prediction_hours}). Repeating values.")
        # Repeat predictions to fill the entire range
        repeat_factor = (num_prediction_hours // len(predictions_df)) + 1
        predictions_df = pd.concat([predictions_df] * repeat_factor)
        predictions_df = predictions_df.iloc[:num_prediction_hours]
    elif len(predictions_df) > num_prediction_hours:
        logger.warning(f"Prediction data has more rows ({len(predictions_df)}) than needed ({num_prediction_hours}). Truncating.")
        predictions_df = predictions_df.iloc[:num_prediction_hours]
    
    # Reindex predictions with the new date range, ignoring the original index
    predictions_reindexed = predictions_df.reset_index(drop=True)
    predictions_reindexed.index = prediction_dates
    
    logger.info(f"Reindexed predictions to hourly data from {predictions_reindexed.index.min()} to {predictions_reindexed.index.max()}")
    
    # Create plots
    plot_country_data(historical_data, predictions_reindexed, spain_columns, "Spain", output_path, team_name)
    plot_country_data(historical_data, predictions_reindexed, italy_columns, "Italy", output_path, team_name)
    plot_total_portfolio(historical_data, predictions_reindexed, spain_columns, italy_columns, output_path, team_name)

def plot_country_data(historical_data, predictions_df, customer_columns, country_name, output_path, team_name):
    """
    Plot data for a specific country
    
    Args:
        historical_data: Historical data DataFrame
        predictions_df: Predictions DataFrame
        customer_columns: List of customer columns to include
        country_name: Name of the country for the plot title
        output_path: Path to save the plots
        team_name: Name of the team for the output file
    """
    if not customer_columns:
        logger.warning(f"No customer columns found for {country_name}")
        return
    
    # Create a total figure for all customers combined
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get the columns that exist in historical data
    historical_cols = [col for col in customer_columns if col in historical_data.columns]
    
    # Plot historical data (total)
    if historical_cols:
        historical_total = historical_data[historical_cols].sum(axis=1)
        if not historical_total.empty:
            historical_total.plot(ax=ax, color='blue', linewidth=2, label='Historical')
            logger.info(f"Plotted historical data for {country_name} with {len(historical_cols)} columns")
    else:
        logger.warning(f"No matching historical columns found for {country_name}")
    
    # Plot prediction data (total)
    prediction_total = predictions_df[customer_columns].sum(axis=1)
    prediction_total.plot(ax=ax, color='red', linewidth=2, linestyle='--', label='Predicted')
    
    # Add a vertical line at the transition point
    transition_date = predictions_df.index.min()
    ax.axvline(x=transition_date, color='black', linestyle='-', linewidth=1, 
               label=f'Prediction Start ({transition_date.strftime("%Y-%m-%d")})')
    
    # Formatting
    ax.set_title(f'{country_name} Total Energy Consumption', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('MWh', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{team_name}_{country_name}_total_prediction.png"), dpi=300)
    plt.close(fig)
    
    # Now create individual customer plots if there are more than one customer
    if len(customer_columns) > 1:
        # Create subplot grid based on number of customers
        n_customers = min(len(customer_columns), 9)  # Limit to 9 plots max for visibility
        n_cols = min(3, n_customers)
        n_rows = (n_customers + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
        if n_rows * n_cols == 1:  # If there's only one subplot
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Use a subset of customers for individual plots to avoid overcrowding
        plot_columns = customer_columns[:n_customers]
        
        for i, column in enumerate(plot_columns):
            if i < len(axes):
                ax = axes[i]
                
                # Extract customer ID from column name for better display
                customer_id = column.split('_')[-1] if '_' in column else column
                
                # Plot historical data
                if column in historical_data.columns:
                    historical_data[column].plot(ax=ax, color='blue', linewidth=2, label='Historical')
                
                # Plot prediction data
                predictions_df[column].plot(ax=ax, color='red', linewidth=2, linestyle='--', label='Predicted')
                
                # Add vertical line at transition
                ax.axvline(x=transition_date, color='black', linestyle='-', linewidth=1)
                
                # Formatting
                ax.set_title(f'Customer {customer_id}', fontsize=12)
                ax.set_ylabel('MWh', fontsize=10)
                ax.legend(fontsize=10)
                ax.grid(True)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        # Add a common x-label
        fig.text(0.5, 0.04, 'Date', ha='center', fontsize=12)
        
        # Format x-axis dates for all subplots
        for ax in axes[:i+1]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'{country_name} Individual Customer Energy Consumption', fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Adjust for the suptitle
        
        plt.savefig(os.path.join(output_path, f"{team_name}_{country_name}_individual_predictions.png"), dpi=300)
        plt.close(fig)

def plot_total_portfolio(historical_data, predictions_df, spain_columns, italy_columns, output_path, team_name):
    """
    Plot total portfolio data (Spain + Italy)
    
    Args:
        historical_data: Historical data DataFrame
        predictions_df: Predictions DataFrame
        spain_columns: List of Spain customer columns
        italy_columns: List of Italy customer columns
        output_path: Path to save the plots
        team_name: Name of the team for the output file
    """
    all_columns = spain_columns + italy_columns
    if not all_columns:
        logger.warning("No customer columns found for the portfolio plot")
        return
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get columns that exist in historical data
    historical_cols = [col for col in all_columns if col in historical_data.columns]
    
    # Plot historical data (total portfolio)
    if historical_cols:
        historical_total = historical_data[historical_cols].sum(axis=1)
        if not historical_total.empty:
            historical_total.plot(ax=ax, color='blue', linewidth=2, label='Historical')
            logger.info(f"Plotted historical data for portfolio with {len(historical_cols)} columns")
    else:
        logger.warning("No matching historical columns found for portfolio")
    
    # Plot prediction data (total portfolio)
    prediction_total = predictions_df[all_columns].sum(axis=1)
    prediction_total.plot(ax=ax, color='red', linewidth=2, linestyle='--', label='Predicted')
    
    # Add a vertical line at the transition point
    transition_date = predictions_df.index.min()
    ax.axvline(x=transition_date, color='black', linestyle='-', linewidth=1, 
               label=f'Prediction Start ({transition_date.strftime("%Y-%m-%d")})')
    
    # Formatting
    ax.set_title('Total Portfolio Energy Consumption (Spain + Italy)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('MWh', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{team_name}_total_portfolio_prediction.png"), dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Plot predictions with historical data')
    parser.add_argument('--dataset', type=str, default='~/Documents/Projects/ACE_Datathon/datasets2025',
                        help='Path to the dataset')
    parser.add_argument('--predictions', type=str, default='~/Documents/Projects/ACE_Datathon/outputs',
                        help='Path to the prediction files')
    parser.add_argument('--output', type=str, default='~/Documents/Projects/ACE_Datathon/outputs/plots',
                        help='Path to save the plots')
    parser.add_argument('--team', type=str, default='Totoro',
                        help='Team name for the output files')
    parser.add_argument('--lookback', type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help='Number of days of historical data to show before predictions (default: 30 days/1 month)')
    
    args = parser.parse_args()
    
    # Expand user paths
    dataset_path = os.path.expanduser(args.dataset)
    predictions_path = os.path.expanduser(args.predictions)
    output_path = os.path.expanduser(args.output)
    
    plot_predictions_with_history(dataset_path, predictions_path, output_path, args.team, args.lookback)

if __name__ == "__main__":
    main()