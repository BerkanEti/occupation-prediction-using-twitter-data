from sklearn.linear_model import LogisticRegression
from models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(max_iter=20000)
        self.name = 'Logistic Regression'
        self.params = {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        }
    
    def get_model_config(self):
        return {
            'model': self.model,
            'name': self.name,
            'params': self.params
        }