# text_processing/utils.py
import re
import emoji
import os
import json
import pandas as pd
from nltk.corpus import stopwords

class DataHandler:
    """Utility class for data operations"""
    
    @staticmethod
    def read_data(data_dir):
        """Read data from JSON files in the specified directory"""
        data = []
        
        for file in os.listdir(data_dir):
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                occupation = file.split('_')[0]
                tweets = json.load(f)
                for tweet in tweets:
                    data.append({
                        'tweet': tweet['tweet'],
                        'source': tweet['source'],
                        'occupation': occupation
                    })
        
        df = pd.DataFrame(data)
        
        # Normalize occupation names
        occupation_mapping = {
            'ogretmen': 'öğretmen',
            'sporyorumcusu': 'spor yorumcusu',
            'tarihci': 'tarihçi',
            'yazilimci': 'yazılımcı',
            'ziraatmuhendisi': 'ziraat mühendisi'
        }
        
        df['occupation'] = df['occupation'].replace(occupation_mapping)
        # get only 0.01 of the data
        df = df.sample(frac=0.01)
        return df
    
    @staticmethod
    def save_to_json(data, output_file, sort_by='occupation'):
        """Save processed data to JSON file"""
        # Convert to DataFrame if it's a list
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
            
        # Sort if specified
        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by)
            
        # Convert to list of dictionaries
        data_list = df.to_dict('records')
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)

class TextPreprocessor:
    """Base class for text preprocessing operations"""
    
    @staticmethod
    def get_stop_words():
        stop_words = set(stopwords.words('turkish'))
        extra_stop_words = {
            'rt', 'via', 'http', 'https', 'www', 'com', 'org', 'html', 'amp', 'retweet',
            've', 'ile', 'da', 'de', 'mi', 'mu', 'mü', 'mı', 'mısın', 'mıyım', 'mısınız', 'misin', 'miyiz', 'mıyız',
            'ya', 'ama', 'çünkü', 'ki', 'oysa', 'hem', 'ya da', 'fakat', 'hatta', 'ancak', 'veya', 'dahi', 
            'için', 'gibi', 'göre', 'kadar', 'karşı', 'sonra', 'önce', 'üzere', 'den', 'dan', 'e', 'a', 'bir', 
            'en', 'bile', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'bu', 'şu', 'bunlar', 'şunlar', 'onun', 
            'benim', 'sizin', 'bizim', 'onların', 'bunu', 'şunu', 'ne', 'kim', 'nerede', 'nasıl', 'niçin', 
            'neden', 'hangi', 'iki', 'üç', 'her', 'bazı', 'tüm', 'hiç', 'çok', 'az', 'daha', 'ise', 'değil', 
            'sadece', 'yalnızca', 'bütün', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz', 'on', 'yirmi', 'otuz',
            'sence', 'bence', 'hem'
        }
        stop_words.update(extra_stop_words)
        return stop_words

    @staticmethod
    def clean_text(text):
        if text is None:
            return ''
            
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove usernames
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Replace hyphens with space
        text = re.sub(r'-', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove escape characters
        text = re.sub(r'\\n|\\t|\\r|\\', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def normalize_turkish_chars(text):
        """Normalize special Turkish characters"""
        text = text.replace('i̇', 'i')
        text = text.replace('î', 'i')
        return text

    @staticmethod
    def filter_short_words(text, min_length=2):
        """Filter out words shorter than min_length"""
        return ' '.join([word for word in text.split() if len(word) > min_length])

# text_processing/zeyrek_processor.py
from zeyrek import MorphAnalyzer

class ZeyrekProcessor:
    """Class for processing text using Zeyrek"""
    
    def __init__(self):
        self.analyzer = MorphAnalyzer()
        self.preprocessor = TextPreprocessor()
        
    def preprocess(self, text):
        """Preprocess text using base preprocessor"""
        stop_words = self.preprocessor.get_stop_words()
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = cleaned_text.split()
        tokens_without_stopwords = [
            token for token in tokens 
            if token not in stop_words and len(token) > 1
        ]
        return ' '.join(tokens_without_stopwords)
        
    def lemmatize(self, text):
        """Lemmatize text using Zeyrek"""
        tokens = text.split()
        lemmatized_tokens = []
        
        for token in tokens:
            analysis = self.analyzer.lemmatize(token)
            if analysis and len(analysis) > 0:
                lemmatized_tokens.append(analysis[0][1][0])
                
        lemmatized_text = ' '.join(lemmatized_tokens)
        lemmatized_text = self.preprocessor.normalize_turkish_chars(lemmatized_text.lower())
        return self.preprocessor.filter_short_words(lemmatized_text)

# text_processing/zemberek_processor.py
from zemberek import (
    TurkishSentenceNormalizer,
    TurkishMorphology,
    TurkishTokenizer
)

class ZemberekProcessor:
    """Class for processing text using Zemberek"""
    
    def __init__(self):
        self.morphology = TurkishMorphology.create_with_defaults()
        self.tokenizer = TurkishTokenizer.DEFAULT
        self.normalizer = TurkishSentenceNormalizer(self.morphology)
        self.preprocessor = TextPreprocessor()
        
    def preprocess(self, text):
        """Preprocess text using base preprocessor"""
        stop_words = self.preprocessor.get_stop_words()
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = cleaned_text.split()
        tokens_without_stopwords = [
            token for token in tokens 
            if token not in stop_words and len(token) > 1
        ]
        return ' '.join(tokens_without_stopwords)
        
    def lemmatize(self, text):
        """Lemmatize text using Zemberek"""
        text = self.normalizer.normalize(text)
        text = self.preprocess(text)
        
        tokens = self.tokenizer.tokenize(text)
        normalized_tokens = []
        
        for token in tokens:
            analysis = self.morphology.analyze_and_disambiguate(token.content)
            if analysis and len(analysis) > 0:
                root = analysis[0].best_analysis.get_stem()
                normalized_tokens.append(root)
            else:
                normalized_tokens.append(token.content)
                
        lemmatized_text = ' '.join(normalized_tokens)
        lemmatized_text = self.preprocessor.normalize_turkish_chars(lemmatized_text.lower())
        return self.preprocessor.filter_short_words(lemmatized_text)

def process_data(data, processor):
    """Process data using specified processor"""
    processed_data = []
    
    for item in data:
        processed_text = processor.preprocess(item['tweet'])
        if processed_text and len(processed_text.split()) > 2:
            lemmatized_text = processor.lemmatize(processed_text)
            if lemmatized_text and len(lemmatized_text.split()) > 3:
                processed_item = {
                    'tweet': item['tweet'],
                    'source': item['source'],
                    'occupation': item['occupation'],
                    'cleaned_tweet': processed_text,
                    'lemmatized_tweet': lemmatized_text
                }
                processed_data.append(processed_item)
    
    return processed_data

# Example usage
if __name__ == "__main__":
    # Initialize data handler and processors
    data_handler = DataHandler()
    zeyrek_processor = ZeyrekProcessor()
    zemberek_processor = ZemberekProcessor()
    
    # Read data
    data = data_handler.read_data('nlp/merged-data')
    
    # Process with Zeyrek
    processed_data_zeyrek = process_data(data.to_dict('records'), zeyrek_processor)
    data_handler.save_to_json(processed_data_zeyrek, 'cleaned-data-zeyrek.json')
    
    # Process with Zemberek
    processed_data_zemberek = process_data(data.to_dict('records'), zemberek_processor)
    data_handler.save_to_json(processed_data_zemberek, 'cleaned-data-zemberek.json')