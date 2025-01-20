from sklearn.ensemble import RandomForestClassifier
from models.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=100)
        self.name = 'Random Forest'
        self.params = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
    
    def get_model_config(self):
        return {
            'model': self.model,
            'name': self.name,
            'params': self.params
        }