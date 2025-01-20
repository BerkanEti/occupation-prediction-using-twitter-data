from sklearn.ensemble import GradientBoostingClassifier
from models.base_model import BaseModel

class GradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier()
        self.name = 'Gradient Boosting'
        self.params = {
            'classifier__n_estimators': [50, 100],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    
    def get_model_config(self):
        return {
            'model': self.model,
            'name': self.name,
            'params': self.params
        }