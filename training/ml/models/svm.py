from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from models.base_model import BaseModel

class SVMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = CalibratedClassifierCV(LinearSVC(max_iter=50000), method='sigmoid', cv=5)
        self.name = 'Calibrated Linear SVM'
        self.params = {
            'classifier__estimator__C': [0.01, 0.1, 1, 10, 100],
            'classifier__estimator__loss': ['hinge', 'squared_hinge'],
            'classifier__estimator__class_weight': ['balanced', None]
        }
    
    def get_model_config(self):
        return {
            'model': self.model,
            'name': self.name,
            'params': self.params
        }