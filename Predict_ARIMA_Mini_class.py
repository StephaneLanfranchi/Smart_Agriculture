from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
import numpy
import csv

series = Series.from_csv('./data/weather_ajaccio.csv', header=0)
split_point = len(series) - 365
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))


def writePredict(model, value):
    with open('./data/predict_result.csv', 'a') as temp_file:
        temp_writer = csv.writer(temp_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        temp_writer.writerow([str(model), value])


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# load dataset
series = Series.from_csv('./data/weather_ajaccio.csv', header=None)
# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7, 0, 1))
model_fit = model.fit(disp=0)
# one-step out-of sample forecast
forecast = model_fit.forecast()[0]
# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)

writePredict("arima_mini", forecast)

print('Forecast: %f' % forecast)