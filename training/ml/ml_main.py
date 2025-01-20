import time

from models.gradient_boosting import GradientBoostingModel
from models.random_forest import RandomForestModel
from models.svm import SVMModel
from models.naive_bayes import NaiveBayesModel
from models.logistic_regression import LogisticRegressionModel

    
from trainer import OccupationTrainer

def get_all_models():
    return [
        LogisticRegressionModel(),
        # RandomForestModel(),
        # NaiveBayesModel(),
        # SVMModel(),
        # GradientBoostingModel()
    ]

def main():
    start_time = time.time()
    
    # Initialize models
    models = get_all_models()
    model_configs = [model.get_model_config() for model in models]
    
    # Run analysis
    file_path = '/Users/berkaneti/Desktop/berkanEti/YTULectures/twitter-meslek-tahmini/berkan_proje/cleaned-data-zeyrek.json'
    predictor = OccupationTrainer(file_path, model_configs)
    results = predictor.run_analysis()
    
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Analiz süresi: {elapsed_time:.2f} dakika")
    
    return results

if __name__ == "__main__":
    main()