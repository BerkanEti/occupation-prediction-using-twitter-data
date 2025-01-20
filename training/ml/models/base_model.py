from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.name = None
        self.params = None
        
    def create_pipeline(self):
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 5),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )),
            ('classifier', self.model)
        ])
    
    @abstractmethod
    def get_model_config(self):
        pass