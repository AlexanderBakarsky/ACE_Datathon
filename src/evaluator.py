from dataloader import Dataset
from tqdm import tqdm

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
    
    