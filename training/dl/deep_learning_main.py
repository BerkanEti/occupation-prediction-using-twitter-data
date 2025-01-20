import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import pickle
from scipy import sparse
import gc


class BaseOccupationPredictor:
    def __init__(self, nlp_method, combination_size, model_type):
        self.nlp_method = nlp_method
        self.combination_size = combination_size
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=30000)
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        file_path = f'cleaned-data-{self.nlp_method}.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    
    def combine_tweets(self, df):
        grouped = df.groupby('occupation')
        combined_data = []
        
        for occupation, group in grouped:
            tweets = group['lemmatized_tweet'].tolist()
            while len(tweets) < self.combination_size:
                tweets.extend(tweets[:self.combination_size - len(tweets)])
            
            for i in range(0, len(tweets), self.combination_size):
                if i + self.combination_size <= len(tweets):
                    combined_tweets = ' '.join(tweets[i:i + self.combination_size])
                    combined_data.append({
                        'occupation': occupation,
                        'combined_tweets': combined_tweets
                    })
            
            if len(combined_data) % 1000 == 0:
                gc.collect()
        
        return pd.DataFrame(combined_data)

    def prepare_data(self):
        print("Loading data...")
        df = self.load_data()
        
        print("Combining tweets...")
        df = self.combine_tweets(df)
        
        print("Vectorizing text...")
        X = self.vectorizer.fit_transform(df['combined_tweets'])
        vocabulary_size = len(self.vectorizer.vocabulary_)
        print(f"Vocabulary size: {vocabulary_size}")
        print(f"Feature matrix shape: {X.shape}")
        
        y = self.label_encoder.fit_transform(df['occupation'])
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def build_model(self, input_dim, num_classes):
        raise NotImplementedError("Subclasses must implement build_model method")

    def train_model(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        num_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]
        
        print("Building model...")
        model = self.build_model(input_dim, num_classes)
        
        save_dir = f'saved_models/trained_{self.nlp_method}_{self.combination_size}_tweets/dl'
        os.makedirs(save_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f'{save_dir}/{self.model_type}.keras',
                save_best_only=True
            )
        ]
        
        print("Training model...")
        history = model.fit(
            X_train.toarray(), y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks
        )
        
        print("Saving vectorizer...")
        vectorizer_path = os.path.join(save_dir, f'vectorizer_{self.model_type}.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        return model, history, X_test, y_test

    def evaluate_and_visualize(self, model, history, X_test, y_test):
        save_dir = f'saved_models/trained_{self.nlp_method}_{self.combination_size}_tweets/dl/visualizations'
        
        batch_size = 1000
        predictions = []
        
        for i in range(0, X_test.shape[0], batch_size):
            batch = X_test[i:i+batch_size].toarray()
            pred_batch = model.predict(batch)
            predictions.append(pred_batch)
            gc.collect()
        
        y_pred_proba = np.vstack(predictions)
        y_pred = np.argmax(y_pred_proba, axis=-1)
        
        n_classes = len(self.label_encoder.classes_)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        self._save_training_plots(history, save_dir)
        self._save_confusion_matrix(y_test, y_pred, save_dir)
        self._save_classification_report(y_test, y_pred, save_dir)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        results = {
            'model_name': f'{self.model_type}_{self.nlp_method}_{self.combination_size}',
            'accuracy': classification_rep['accuracy'],
            'precision': np.mean([v['precision'] for k, v in classification_rep.items() if k not in ['accuracy', 'macro avg', 'weighted avg']]),
            'recall': np.mean([v['recall'] for k, v in classification_rep.items() if k not in ['accuracy', 'macro avg', 'weighted avg']]),
            'f1_score': np.mean([v['f1-score'] for k, v in classification_rep.items() if k not in ['accuracy', 'macro avg', 'weighted avg']]),
            'confusion_matrix': conf_matrix.tolist(),
            'history': history.history,
            'y_test_bin': y_test_bin.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        }
        
        self._save_results(results, save_dir)
        return results
    def _save_results(self, results, save_dir):
        results_file = os.path.join(save_dir, f'{self.model_type}_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def _save_training_plots(self, history, save_dir):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'{self.model_type} Model Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'{self.model_type} Model Loss')
        plt.legend()
        plt.savefig(f'{save_dir}/{self.model_type}_training_history.png')
        plt.close()
    
    def _save_confusion_matrix(self, y_test, y_pred, save_dir):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(f'{self.model_type} Confusion Matrix')
        plt.savefig(f'{save_dir}/{self.model_type}_confusion_matrix.png')
        plt.close()
        
    def _save_classification_report(self, y_test, y_pred, save_dir):
        report = classification_report(
            y_test,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        with open(f'{save_dir}/{self.model_type}_classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
            
    def advanced_visualization(self, results_list):
        save_dir = f'saved_models/trained_{self.nlp_method}_{self.combination_size}_tweets/dl/visualizations'
        


        n_models = len(results_list)
        n_cols = 2
        n_rows = (n_models + 1) // n_cols
        
        plt.figure(figsize=(15, 10))
        for idx, result in enumerate(results_list, 1):
            plt.subplot(n_rows, n_cols, idx)
            
            y_test_bin = np.array(result['y_test_bin'])
            y_pred_proba = np.array(result['y_pred_proba'])
            
            n_classes = y_test_bin.shape[1]
            
            colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
            
            for i, color in zip(range(n_classes), colors):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = self.label_encoder.classes_[i]
                plt.plot(fpr, tpr, color=color, 
                        label=f'{class_name}\n(AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.title(f'{result["model_name"]}\nROC Curves')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right', bbox_to_anchor=(1.7, 0))
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()

        plt.figure(figsize=(10, 8))
        
        for result in results_list:
            y_test_bin = np.array(result['y_test_bin'])
            y_pred_proba = np.array(result['y_pred_proba'])
            n_classes = y_test_bin.shape[1]
            
            all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], 
                              y_pred_proba[:, i])[0] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                mean_tpr += np.interp(all_fpr, fpr, tpr)
            
            mean_tpr /= n_classes
            macro_roc_auc = auc(all_fpr, mean_tpr)
            
            plt.plot(all_fpr, mean_tpr,
                    label=f'{result["model_name"]} (AUC = {macro_roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Macro-average ROC Curves')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'macro_roc_curves.png'), dpi=300)
        plt.close()

class CNNOccupationPredictor(BaseOccupationPredictor):
    def __init__(self, nlp_method, combination_size):
        super().__init__(nlp_method, combination_size, "CNN")
    
    def build_model(self, input_dim, num_classes):
        model = Sequential([
            Reshape((input_dim, 1), input_shape=(input_dim,)),
            
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Flatten(),
            
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

class MLPOccupationPredictor(BaseOccupationPredictor):
    def __init__(self, nlp_method, combination_size):
        super().__init__(nlp_method, combination_size, "MLP")
    
    def build_model(self, input_dim, num_classes):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model

def train_all_configurations():
    nlp_methods = ['zeyrek', 'zemberek']
    combination_sizes = [1, 2, 3, 5]
    all_results = []
    
    for nlp_method in nlp_methods:
        method_results = []
        for combination_size in combination_sizes:
            print(f"\nTraining models for {nlp_method} with {combination_size} tweets combination")
            
            try:
                print("\nTraining CNN model...")
                cnn_predictor = CNNOccupationPredictor(nlp_method, combination_size)
                cnn_model, cnn_history, X_test, y_test = cnn_predictor.train_model()
                cnn_results = cnn_predictor.evaluate_and_visualize(cnn_model, cnn_history, X_test, y_test)
                method_results.append(cnn_results)
                
                gc.collect()
                
                print("\nTraining MLP model...")
                mlp_predictor = MLPOccupationPredictor(nlp_method, combination_size)
                mlp_model, mlp_history, X_test, y_test = mlp_predictor.train_model()
                mlp_results = mlp_predictor.evaluate_and_visualize(mlp_model, mlp_history, X_test, y_test)
                method_results.append(mlp_results)
                
                cnn_predictor.advanced_visualization([cnn_results, mlp_results])
                
                gc.collect()
                
            except Exception as e:
                print(f"Error training models for {nlp_method} with {combination_size} tweets: {str(e)}")
                continue
        
        if method_results:
            results_file = f'results_{nlp_method}.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(method_results, f, ensure_ascii=False, indent=4)
            all_results.extend(method_results)
    
    if all_results:
        with open('all_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    train_all_configurations()