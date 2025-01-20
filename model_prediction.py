import tensorflow as tf
import joblib
import os
import json
import re
from nltk.corpus import stopwords
import nltk
import emoji
import pickle
from zemberek import (
        TurkishSentenceNormalizer, TurkishMorphology, TurkishTokenizer
    )
import numpy as np

morphology = TurkishMorphology.create_with_defaults()
tokenizer = TurkishTokenizer.DEFAULT
normalizer = TurkishSentenceNormalizer(morphology)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class Utils:
    def get_stop_words(self):
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
        
        return stop_words


    def clean_text(self,text):
        if text is None:
            return ''
        text = emoji.replace_emoji(text, replace='')
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'@\w+', '', text)
        
        text = re.sub(r'#\w+', '', text)

        text = re.sub(r'-', ' ', text)
        
        text = re.sub(r'[^\w\s]', ' ', text)

        text = re.sub(r'\\n|\\t|\\r|\\', ' ', text)
        
        text = text.lower()

        text = re.sub(r'\d+', '', text)

        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess(self,text):
        stopwords = self.get_stop_words()
        
        cleaned_text = self.clean_text(text)
        
        tokens = cleaned_text.split()
        
      
        tokens_without_stopwords = [
            token for token in tokens 
            if token not in stopwords and len(token) > 1
        ]

        return ' '.join(tokens_without_stopwords)

    def lemmatize(self,text):

        text=normalizer.normalize(text)

        normalized_tokens = []
        text = self.preprocess(text)

        tokens = tokenizer.tokenize(text)
    
        for token in tokens:
    
            analysis = morphology.analyze_and_disambiguate(token.content)
            if analysis and len(analysis) > 0:
                root = analysis[0].best_analysis.get_stem()
                normalized_tokens.append(root)
            else:
                normalized_tokens.append(token.content)
    
        
        return ' '.join(normalized_tokens)


class OccupationPredictorML:
    def __init__(self, models_dir):
        self.utils = Utils()
        self.models_dir = models_dir
        self.models = {}
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
        
        encoder_path = os.path.join(self.models_dir, 'label_encoder.joblib')
        self.label_encoder = joblib.load(encoder_path)
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib') and filename != 'label_encoder.joblib':
                model_name = filename.replace('.joblib', '')
                model_path = os.path.join(self.models_dir, filename)
                self.models[model_name] = joblib.load(model_path)
            

    def predict(self, text, model_name=None):
        
        processed_text = self.utils.lemmatize(text)
        
        results = {}
        if model_name and model_name in self.models:
            models_to_use = {model_name: self.models[model_name]}
        else:
            models_to_use = self.models
        
        for name, model in models_to_use.items():
            prediction = model.predict([processed_text])[0]
            predicted_class = self.label_encoder.inverse_transform([prediction])[0]
            
            

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([processed_text])[0]
                prob_dict = dict(zip(self.label_encoder.classes_, probabilities))
                prob_dict = {k: v * 100 for k, v in prob_dict.items()}
                results[name] = {
                    'prediction': predicted_class,
                    'probabilities': prob_dict,
                    'confidence': probabilities[prediction] * 100
                }
            else:
                results[name] = {
                    'prediction': predicted_class
                }
        
        return results

class OccupationPredictorDL:
    def __init__(self, models_path):
        self.utils = Utils()
        self.models = {}
        self.vectorizers = {}
        self.label_encoder = None
        self.load_models_and_vectorizers(models_path)
        
    def load_models_and_vectorizers(self, models_path):
        encoder_path = os.path.join(models_path, 'label_encoder.joblib')
        self.label_encoder = joblib.load(encoder_path)

        for filename in os.listdir(models_path):
            if filename.endswith('.keras'):
    
                model_name = filename.replace('.keras', '')
                model_path = os.path.join(models_path, filename)
                self.models[model_name] = tf.keras.models.load_model(model_path)
                
        for filename in os.listdir(models_path):
            if filename.endswith('.pkl'):
    
                vectorizer_name = filename.replace('.pkl', '').replace('vectorizer_', '')
                vectorizer_path = os.path.join(models_path, filename)
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizers[vectorizer_name] = pickle.load(f)
        
    def predict(self, text, model_name=None):
        processed_text = self.utils.lemmatize(text)
   
        results = {}
        if model_name and model_name in self.models:
            models_to_use = {model_name: self.models[model_name]}
        else:
            models_to_use = self.models
        
        for name, model in models_to_use.items():

            vectorizer = self.vectorizers.get(name)
            if vectorizer is None:
                raise KeyError(f"No vectorizer found for model {name}. Available vectorizers: {list(self.vectorizers.keys())}")
                
            text_vector = vectorizer.transform([processed_text]).toarray()
            y_pred_prob = model.predict(text_vector)
            y_pred_class = np.argmax(y_pred_prob, axis=1)
            occupation = self.label_encoder.inverse_transform(y_pred_class)[0]
            confidence = np.max(y_pred_prob) * 100
            all_predictions = y_pred_prob[0] * 100
            results[name] = {
                'prediction': occupation,
                'confidence': confidence,
                'probabilities': dict(zip(self.label_encoder.classes_, all_predictions))
            }
        
        return results

def main():
    models_dir = 'saved_models/trained_zemberek_1_tweets/dl/'  
    predictor = OccupationPredictorDL(models_dir)

    print("Available models:", predictor.models.keys())
    print("Available vectorizers:", predictor.vectorizers.keys())
    
    text = "Merhaba ben Berkan Eti. kod yazıyorum"
    results = predictor.predict(text)
    print("Prediction results:", results)

if __name__ == "__main__":
    main()

