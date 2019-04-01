import numpy as np
import pandas as pa
import csv
from statsmodels.tsa.arima_model import ARIMA
from pythonpdevs.src.Utils import Timer
from sklearn.metrics import mean_squared_error

# get data
def GetData(fileName):
    return pa.read_csv(fileName, header=0, parse_dates=[0], index_col=0)


def trainingSet(serie):
    trainPercent = 0.90
    X = serie.values
    size = int(len(X) * trainPercent)
    train = X[0:size]
    test = X[size:len(X)]
    return train, test


def startTraining(filename):
    series = GetData(filename)
    predictions = list()
    train, test = trainingSet(series)
    history = [x for x in train]
    for t in range(len(test)):
        model = ARIMA(history, order=(7, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #print('predicted=%f, expected=%f' % (int(yhat), obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)

    X = difference(series.values)

    # save coefficients
    coef = model_fit.params
    window_size = 7
    np.save('./data/model/temp_model.npy', coef)

    # save lag
    lag = X[-window_size:]
    np.save('./data/temp_data.npy', lag)

    # save the last ob
    np.save('./data/temp_obs.npy', [series.values[-1]])

    # Start temperature prediction
    #predictions = startPredict()

    #return predictions


# create a difference transform of the dataset
def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)


def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


def startPredict():
    timer = Timer()
    timer.start()
    # load model
    coef = np.load('./data/model/temp_model.npy')
    lag = np.load('./data/temp_data.npy')
    last_ob = np.load('./data/temp_obs.npy')

    # make prediction
    prediction = predict(coef, lag)
    # transform prediction
    yhat = prediction + last_ob[0]
    timer.stop()
    print('Prediction: %f' % yhat)

    return yhat
