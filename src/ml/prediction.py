def predict(model, input_data):
    """
    Make predictions using the trained machine learning model.

    Parameters:
    model: The trained machine learning model.
    input_data: A list or array of input features for prediction.

    Returns:
    predictions: The predicted output from the model.
    """
    predictions = model.predict(input_data)
    return predictions

def load_model(model_path):
    """
    Load a trained machine learning model from a specified path.

    Parameters:
    model_path: The file path to the trained model.

    Returns:
    model: The loaded machine learning model.
    """
    import joblib
    model = joblib.load(model_path)
    return model

def save_model(model, model_path):
    """
    Save the trained machine learning model to a specified path.

    Parameters:
    model: The machine learning model to save.
    model_path: The file path where the model will be saved.
    """
    import joblib
    joblib.dump(model, model_path)