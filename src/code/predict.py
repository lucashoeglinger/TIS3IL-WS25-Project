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

    return None

def predict_ml(enc_length, path_to_pkl="../pkl"):

    return None


