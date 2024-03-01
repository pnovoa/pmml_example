from flask import Flask
# import joblib
from sklearn_pmml_model.auto_detect import auto_detect_classifier


# Initialize App
app = Flask(__name__)

# Load models
# model = joblib.load('model/model_binary.dat.gz')

model = auto_detect_classifier(pmml="model/decision_tree.pmml")