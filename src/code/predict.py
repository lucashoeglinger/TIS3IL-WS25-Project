from neuralforecast import NeuralForecast
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals

import glob
import os
import statsmodels.api as sm
import numpy as np

def predict_neural(enc_length, model_paths="../ckpt", level=[95]):
    """
    Predicts future values and prediction intervals using NeuralForecast models.

    -------Params-------
    enc_length int: The number of steps to predict in the future.
    model_paths str: Path to saved NeuralForecast models.
    level list[int]: Confidence levels (e.g., [95] for a 95% interval).
    """
    nf = NeuralForecast.load(path=model_paths, allow_pickle=True)

    YHat = nf.predict(h=enc_length, level=level)

    return YHat


def predict_stats(enc_length, path_to_pkl="../pkl", alpha=0.05):
    """
    Makes predictions and calculates confidence intervals for statsmodels models.

    -----Params------
    enc_length int: The number of steps to predict into the future.
    path_to_pkl str: Path to all `.pkl` files.
    alpha float: Significance level (0.05 = 95% confidence interval).

    -----Returns-----
    results dict: Dictionary mapping model names to tuples of (forecast, conf_int).
    """
    results = {}
    for path in glob.glob(os.path.join(path_to_pkl, "*.pkl")):
        name = os.path.splitext(os.path.basename(path))[0]
        model = sm.load(path)

        fc_obj = model.get_forecast(steps=enc_length)
        fc = fc_obj.predicted_mean
        ci = fc_obj.conf_int(alpha=alpha)
        
        results[name] = (fc, ci)

    return results


def predict_ml(enc_length, path_to_pkl="../pkl", level=[95]):
    """
    Makes predictions and calculates prediction intervals for MLForecast models.

    -----Params------
    enc_length int: The number of steps to predict into the future.
    path_to_pkl str: Path to saved MLForecast directory.
    level list[int]: Confidence levels (e.g., [95] for a 95% interval).
    """
    mlf = MLForecast.load(path_to_pkl)

    Y_HAT = mlf.predict(h=enc_length, level=level)

    return Y_HAT