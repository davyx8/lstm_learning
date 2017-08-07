import pandas as pd
import numpy as np
import keras as kr
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('data/international-airline-passengers.csv',usecols=[1],skipfooter=3)
print data
# plt.plot(data)
# plt.show

vals = data.values.astype('float32')
print vals.mean()
dataset=vals
look_back=1
train_size= int(len(vals)*0.67)
test_size=  len(vals) - train_size
test_set = vals[train_size:,:]
train_set = vals[0:train_size,:]



def shift(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)




train_x, train_y = shift(train_set,look_back)
test_x , test_y = shift(test_set,look_back)
# model creation
model = Sequential()
model.add(Dense(output_dim=8,input_dim=1,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(train_x,train_y,batch_size=2,verbose=0,nb_epoch=300)
test_score = model.evaluate(test_x,test_y,verbose=0)
print 'test score is {}'.format(test_score)
print 'test score sqrt  is {}'.format(math.sqrt(test_score))
train_score = model.evaluate(train_x,train_y,verbose=1)
print 'train score is {}'.format(train_score)
print 'train score sqrt  is {}'.format(math.sqrt(train_score))

train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

'''
# generate predictions for training
trainPredict = model.predict(train_x)
test_predict = model.predict(test_x)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = test_predict
# plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()'''
