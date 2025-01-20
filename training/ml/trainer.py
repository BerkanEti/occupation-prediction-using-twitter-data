import json
import numpy as np
import pandas as pd
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import parallel_backend
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


METHOD = 'version_1'

class OccupationTrainer:
    def __init__(self, file_path, model_configs):
        self.df = self.load_data(file_path)
        self.df['processed_tweet'] = self.df['lemmatized_tweet'].apply(self.advanced_preprocessing)
        self.df = self.combine_tweets(self.df)
        self.label_encoder = LabelEncoder()
        self.df['occupation_encoded'] = self.label_encoder.fit_transform(self.df['occupation'])
        self.models = model_configs
        self.X = self.df['processed_tweet']
        self.y = self.df['occupation_encoded']
        self.occupation_labels = list(self.label_encoder.classes_)

    
    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        df = pd.DataFrame(data)
        df.drop_duplicates(subset=['lemmatized_tweet', 'occupation'], inplace=True)

        return df
    
    def combine_tweets(self, df, combination_size=10):
        grouped = df.groupby('occupation')
        combined_data = []
        
        for occupation, group in grouped:
            tweets = group['processed_tweet'].tolist()
            while len(tweets) < combination_size:
                tweets.extend(tweets[:combination_size - len(tweets)])
            
            for i in range(0, len(tweets), combination_size):
                if i + combination_size <= len(tweets):
                    combined_tweets = ' '.join(tweets[i:i + combination_size])
                    
                    combined_data.append({
                        'occupation': occupation,
                        'processed_tweet': combined_tweets
                    })
        
        return pd.DataFrame(combined_data)
    
    def advanced_preprocessing(self, text):
       
        text = text.lower()
        text = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        
        tokens = word_tokenize(text)
        
        stop_words = set(stopwords.words('turkish'))
        extra_stop_words = {
            'rt', 'via', 'http', 'https', 'www', 'com', 'org', 'html', 'amp', 'retweet',
            've', 'ile', 'da', 'de', 'mi', 'mu', 'mü', 'mı', 'mısın', 'mıyım', 'mısınız', 'misin', 'miyiz', 'mıyız',
            'ya', 'ama', 'çünkü', 'ki', 'oysa', 'hem', 'ya da', 'fakat', 'hatta', 'ancak', 'veya', 'dahi', 
            'için', 'gibi', 'göre', 'kadar', 'karşı', 'sonra', 'önce', 'üzere', 'den', 'dan', 'e', 'a', 'bir', 
            'en', 'bile', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'bu', 'şu', 'bunlar', 'şunlar', 'onun', 
            'benim', 'sizin', 'bizim', 'onların', 'bunu', 'şunu', 'ne', 'kim', 'nerede', 'nasıl', 'niçin', 
            'neden', 'hangi', 'iki', 'üç', 'her', 'bazı', 'tüm', 'hiç', 'çok', 'az', 'daha', 'ise', 'değil', 
            'sadece', 'yalnızca', 'bütün','dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on', 'yirmi', 'otuz','sence','bence','hem'
        }
        
        stop_words.update(extra_stop_words)
        tokens = [token for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def create_pipeline(self, model):
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=2000,  
                ngram_range=(1, 2),  
                min_df=2,  
                max_df=0.95,  
                sublinear_tf=True
            )),
            ('classifier', model)
        ])
    
    def evaluate_model(self, model_config):
        print(f"Evaluating model: {model_config['name']}")
        model = model_config['model']
        model_name = model_config['name']
        
        
        
        stratified_cv = StratifiedKFold(
            n_splits=10,  
            shuffle=True, 
            random_state=42
        )
        
        
        pipeline = self.create_pipeline(model)
        
        
        print(f"Running Grid Search for {model_name}")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid=model_config['params'], 
            cv=stratified_cv, 
            scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
            refit='f1_weighted',
            n_jobs=-1 
        )
        print(f"Splitting data for {model_name}")
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
       
        with parallel_backend('multiprocessing'):
            grid_search.fit(X_train, y_train)
        
        
        best_model = grid_search.best_estimator_
        
        
        y_pred = best_model.predict(X_test)
        
        y_pred_proba = best_model.predict_proba(X_test)
        
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        #classifaction report
        print(f"Model: {model_name}")
        print(classification_report(y_test, y_pred, target_names=self.occupation_labels))

        
        y_test_bin = label_binarize(y_test, classes=np.unique(self.y))
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'best_params': grid_search.best_params_,
            'y_test_bin': y_test_bin,
            'y_pred_proba': y_pred_proba,
            'best_model': best_model
        }
    
    def advanced_visualization(self, results):
        
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'f1_score']
        performance_data = []
        labels = []
        for result in results:
            performance_data.append([result[metric] for metric in metrics])
            labels.append(result['model_name'])
        
        performance_df = pd.DataFrame(
            performance_data, 
            columns=metrics, 
            index=labels
        )
        
        performance_df.plot(kind='bar', rot=45, cmap='viridis')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.tight_layout()
        
        plt.savefig(f"model_performance_comparison_{METHOD}.png")
        plt.show()

        
        plt.figure(figsize=(15, 10))
        for idx, result in enumerate(results, 1):
            plt.subplot(2, 3, idx)
            sns.heatmap(
                result['confusion_matrix'], 
                annot=True, 
                fmt='.0f', 
                cmap='YlGnBu',
                xticklabels=self.occupation_labels,
                yticklabels=self.occupation_labels
            )
            plt.title(f'{result["model_name"]} - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{METHOD}.png")
        plt.show()

        
        plt.figure(figsize=(15, 10))
        for idx, result in enumerate(results, 1):
            plt.subplot(2, 3, idx)
            y_test_bin = result['y_test_bin']
            y_pred_proba = result['y_pred_proba']
            
            for i in range(len(np.unique(self.y))):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'{result["model_name"]} - ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"roc_curve_{METHOD}.png")
        plt.show()
    
    def save_results_to_json(self, results):
        results_to_save = []
        for result in results:
            result_dict = {
                'model_name': result['model_name'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'best_params': str(result['best_params'])
            }
            results_to_save.append(result_dict)
        
        
        json_file_name = f"model_performance_results_{METHOD}.json"
        with open(json_file_name, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)

    def save_trained_models(self, results, save_dir='trained'): 
        save_dir = f"{save_dir}_{METHOD}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        
        for result in results:
            model_name = result['model_name'].replace(' ', '_').lower()
            
            
            model_path = os.path.join(save_dir, f"{model_name}.joblib")
            joblib.dump(result['best_model'], model_path)
            
            
            encoder_path = os.path.join(save_dir, 'label_encoder.joblib')
            joblib.dump(self.label_encoder, encoder_path)
            
            
            metrics = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'best_params': str(result['best_params']),
            }

            #confusion matrix'i foto olarak kaydet
            plt.figure(figsize=(15, 10))
            sns.heatmap(
                result['confusion_matrix'], 
                annot=True, 
                fmt='.0f', 
                cmap='YlGnBu',
                xticklabels=self.occupation_labels,
                yticklabels=self.occupation_labels
            )
            plt.title(f'{result["model_name"]} - Confusion Matrix')
            plt.tight_layout()
            cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(cm_path)

            
            plt.figure(figsize=(15, 10))
            y_test_bin = result['y_test_bin']
            y_pred_proba = result['y_pred_proba']

            for i in range(len(np.unique(self.y))):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f}')
                
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f'{result["model_name"]} - ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
            plt.savefig(roc_path)
            
            metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
        
        print(f"Tüm modeller ve ilgili bileşenler '{save_dir}' klasörüne kaydedildi.")
    
    def run_analysis(self):
        print("running analysis")
        results = []
        for model_config in self.models:
            result = self.evaluate_model(model_config)
            results.append(result)
        
        
        self.save_trained_models(results)
        
        
        self.advanced_visualization(results)
        
        
        self.save_results_to_json(results)
        
        return results