from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

file_paths = ["./data-1-prepped/aggregated_data.csv", "./data-2-prepped/aggregated_data.csv",
              "./data-3-prepped/aggregated_data.csv", "./data-4-prepped/aggregated_data.csv"]
dfs = []

for fpath in file_paths:
    df = pd.read_csv(fpath, index_col=None)
    df['isDistracted'] = df.pop('isDistracted')
    df.fillna(-1, inplace=True)
    dfs.append(df)

dataset = pd.concat(dfs)
print(dataset)
# load dataset
# dataset = pd.read_csv("./data-1-prepped/aggregated_data.csv", header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# # drop columns we don't want to predict
# reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed[:5])
reframed.to_csv('sample.csv')
 
# split into train and test sets
values = reframed.values
n_train = int(reframed.shape[0] * 0.8)
train = values[:n_train, :]
test = values[n_train:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
 
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

full_inv_yhat = np.zeros((yhat.shape[0], scaled.shape[1]))
full_inv_yhat[:, -1] = yhat[:, 0]
inv_yhat = scaler.inverse_transform(full_inv_yhat)
inv_yhat = inv_yhat[:, -1]
print(inv_yhat)

test_y = test_y.reshape((len(test_y), 1))
full_inv_y = np.zeros((test_y.shape[0], scaled.shape[1]))
full_inv_y[:, -1] = test_y[:, 0]
inv_y = scaler.inverse_transform(full_inv_y)
inv_y = inv_y[:, -1]

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

threshold = 0.5
predictions = (inv_yhat >= threshold).astype(int)
actual = (inv_y >= threshold).astype(int)

accuracy = accuracy_score(actual, predictions)
precision = precision_score(actual, predictions)
print('Test Accuracy: %.3f' % accuracy)
print('Test Precision: %.3f' % precision)