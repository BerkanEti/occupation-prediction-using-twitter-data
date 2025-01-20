from sklearn.naive_bayes import MultinomialNB
from models.base_model import BaseModel

class NaiveBayesModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = MultinomialNB()
        self.name = 'Multinomial Naive Bayes'
        self.params = {
            'classifier__alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        }
    
    def get_model_config(self):
        return {
            'model': self.model,
            'name': self.name,
            'params': self.params
        }