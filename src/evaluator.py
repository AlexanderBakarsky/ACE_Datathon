from dataloader import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self,
            model,
            data_path,
        ):
        self.model = model
        self.dataset = Dataset(data_path)
        
    def evaluate(self,
                 split_index: int,
                 ):
        train_data, input_data, target_data = self.dataset[split_index]
        
        target_columns = target_data.columns
        target_columns = [col for col in target_columns if 'VALUEMWHMETERINGDATA' in col]
        
        output = self.model(train_data, input_data, target_columns)
        
        # calculate the absolute error
        spain_columns = [col for col in target_data.columns if 'VALUEMWHMETERINGDATA_customerES' in col]
        italy_columns = [col for col in target_data.columns if 'VALUEMWHMETERINGDATA_customerIT' in col]
        # spain results
        
        spain_target = target_data[spain_columns]
        spain_output = output[spain_columns]
        
        spain_error = (spain_target - spain_output).abs().sum().sum()
        spain_error_portfolio = (spain_target - spain_output).sum(axis=1).abs().sum()
        
        # italy results
        
        italy_target = target_data[italy_columns]
        italy_output = output[italy_columns]
        
        italy_error = (italy_target - italy_output).abs().sum().sum()
        italy_error_portfolio = (italy_target - italy_output).sum(axis=1).abs().sum()
        
        forecast_score = (
            1.0 * italy_error + 5.0 * spain_error + 10.0 * italy_error_portfolio + 50.0 * spain_error_portfolio
        )
        
        return forecast_score

    def evaluate_all(self, verbose=True):
        forecast_scores = []
        for split_index in tqdm(range(len(self.dataset)), desc="Evaluating all splits", disable=not verbose):
            forecast_scores.append(self.evaluate(split_index))
            logger.info(f"Forecast score for split {split_index}: {forecast_scores[-1]}")
            
        mean_forecast_score = sum(forecast_scores) / len(forecast_scores)
        logger.info(f"Mean forecast score: {mean_forecast_score}")
        return mean_forecast_score, forecast_scores

    def plot_predictions(self, 
                         split_index: int,
                         save_path: str = None,
                         show_plot: bool = True):
        """
        Plot the predicted output with the actual output for a specific split.
        
        Args:
            split_index (int): The index of the split to plot
            save_path (str, optional): Path to save the plot. If None, plot is not saved.
            show_plot (bool, optional): Whether to display the plot. Defaults to True.
            
        Returns:
            tuple: Figure and axes objects for further customization if needed
        """
        train_data, input_data, target_data = self.dataset[split_index]
        
        target_columns = target_data.columns
        target_columns = [col for col in target_columns if 'VALUEMWHMETERINGDATA' in col]
        
        output = self.model(train_data, input_data, target_columns)
        
        # Ensure DataFrame indices are aligned for comparison
        target_data = target_data.set_index('DATETIME') if 'DATETIME' in target_data.columns else target_data
        output = output.set_index('DATETIME') if 'DATETIME' in output.columns else output
        
        # Make sure indices are properly converted to datetime
        if not pd.api.types.is_datetime64_any_dtype(target_data.index):
            logger.warning("Target data index is not datetime64. Converting...")
            target_data.index = pd.to_datetime(target_data.index)
        
        if not pd.api.types.is_datetime64_any_dtype(output.index):
            logger.warning("Output data index is not datetime64. Converting...")
            output.index = pd.to_datetime(output.index)
        
        # Split into Spain and Italy columns
        spain_columns = [col for col in target_columns if 'customerES' in col]
        italy_columns = [col for col in target_columns if 'customerIT' in col]
        
        # Create a multi-panel plot: one for Spain, one for Italy, and one for portfolio totals
        fig, axes = plt.subplots(3, 1, figsize=(15, 18), sharex=True)
        
        # Plot Spain predictions vs actuals
        if spain_columns:
            spain_target = target_data[spain_columns].sum(axis=1)
            spain_pred = output[spain_columns].sum(axis=1)
            
            # Use matplotlib directly for plotting to avoid pandas index type issues
            axes[0].plot(spain_target.index, spain_target.values, label='Actual', linewidth=2)
            axes[0].plot(spain_pred.index, spain_pred.values, label='Predicted', 
                         linewidth=2, linestyle='--')
            
            axes[0].set_title(f'Spain Total Energy Consumption - Split {split_index}')
            axes[0].set_ylabel('MWh')
            axes[0].legend()
            axes[0].grid(True)
            
            # Format x-axis dates
            axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            axes[0].xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=1))
            plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
            
            # Log some debugging info about the data
            logger.info(f"Spain target: min={spain_target.min()}, max={spain_target.max()}, mean={spain_target.mean()}")
            logger.info(f"Spain prediction: min={spain_pred.min()}, max={spain_pred.max()}, mean={spain_pred.mean()}")
        
        # Plot Italy predictions vs actuals
        if italy_columns:
            italy_target = target_data[italy_columns].sum(axis=1)
            italy_pred = output[italy_columns].sum(axis=1)
            
            # Use matplotlib directly for plotting to avoid pandas index type issues
            axes[1].plot(italy_target.index, italy_target.values, label='Actual', linewidth=2)
            axes[1].plot(italy_pred.index, italy_pred.values, label='Predicted', 
                        linewidth=2, linestyle='--')
            
            axes[1].set_title(f'Italy Total Energy Consumption - Split {split_index}')
            axes[1].set_ylabel('MWh')
            axes[1].legend()
            axes[1].grid(True)
            
            # Format x-axis dates
            axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            axes[1].xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=1))
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
            
            # Log some debugging info about the data
            logger.info(f"Italy target: min={italy_target.min()}, max={italy_target.max()}, mean={italy_target.mean()}")
            logger.info(f"Italy prediction: min={italy_pred.min()}, max={italy_pred.max()}, mean={italy_pred.mean()}")
        
        # Plot total portfolio
        total_target = target_data[target_columns].sum(axis=1)
        total_pred = output[target_columns].sum(axis=1)
        
        # Use matplotlib directly for plotting to avoid pandas index type issues
        axes[2].plot(total_target.index, total_target.values, label='Actual', linewidth=2)
        axes[2].plot(total_pred.index, total_pred.values, label='Predicted', 
                    linewidth=2, linestyle='--')
        
        axes[2].set_title(f'Total Portfolio Energy Consumption - Split {split_index}')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('MWh')
        axes[2].legend()
        axes[2].grid(True)
        
        # Format x-axis dates
        axes[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        axes[2].xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=1))
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        
        # Log some debugging info about the data
        logger.info(f"Total target: min={total_target.min()}, max={total_target.max()}, mean={total_target.mean()}")
        logger.info(f"Total prediction: min={total_pred.min()}, max={total_pred.max()}, mean={total_pred.mean()}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, axes
    
    def plot_all_predictions(self, 
                           save_dir: str = None,
                           show_plots: bool = False):
        """
        Plot predictions for all splits
        
        Args:
            save_dir (str, optional): Directory to save plots. If None, plots are not saved.
            show_plots (bool, optional): Whether to display each plot. Defaults to False.
            
        Returns:
            list: List of (fig, axes) tuples for all plots
        """
        plots = []
        for split_index in tqdm(range(len(self.dataset)), desc="Plotting all splits"):
            save_path = None
            if save_dir:
                import os
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir, f"split_{split_index}_prediction.png")
            
            fig, axes = self.plot_predictions(
                split_index=split_index,
                save_path=save_path,
                show_plot=show_plots
            )
            plots.append((fig, axes))
            
            if not show_plots:
                plt.close(fig)
                
        return plots

