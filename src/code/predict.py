from neuralforecast import NeuralForecast

def predict_neural(enc_length, model_paths="../ckpt"):
    """
    This Model is used to predict the neural models of this package

    -------Params-------
    enc_length int: The values to predict in the future
    model_paths str: Path to all trained model paths (use root directory)
    """

    nf = NeuralForecast.load(path=model_paths, allow_pickle=True)
    YHat = nf.predict(h=enc_length)

    return YHat

def predict_stats(enc_length, path_to_pkl="../pkl"):

    """
    This Function is used to make predictions rom the statsmodels of the Weather pred system.

    -----Params------
    enc_length `int`: The values to predict into the future
    path_to_pkl `os.Path`: Path to all `.pkl` files.
    """

    return None

def predict_ml(enc_length, path_to_pkl="../pkl"):
    """
    This Function is used to make predictions rom the Machine learning models of the Weather pred system.

    -----Params------
    enc_length `int`: The values to predict into the future
    path_to_pkl `os.Path`: Path to all `.pkl` files.
    """
        
    return None


