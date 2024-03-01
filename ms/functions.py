import pandas as pd
from ms import model


def get_model_prediction(json_data):
    X = pd.DataFrame.from_dict(json_data)
    prediction = model.predict(X)
    return {
        'status': 200,
        'label': str(list(prediction))
    }