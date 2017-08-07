import pandas as pd
import numpy as np
import keras as kr
# import matplotlib.pyplot as plt
import pandas
import math
import pdb
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dense

data= pd.read_csv('data/international-airline-passengers.csv',usecols=[1],skipfooter=3)
# plt.plot(data)
# plt.show

vals = data.values.astype('float32')

dataset=vals
lookback=3
scaler = MinMaxScaler(feature_range=(0, 1))
vals = scaler.fit_transform(vals)
train_size= int(len(vals)*0.67)
test_size=  len(vals) - train_size
test_set = vals[train_size:,:]
train_set = vals[0:train_size,:]
def shift(data,lookback=1):
    data_x , data_y= [],[]
    for i in xrange(len(data)-lookback-1):
        a = data[i:i+lookback,0]
        data_x.append(a)
        data_y.append( data[i+lookback , 0] )
    return np.array(data_x),np.array(data_y)

train_x, train_y = shift(train_set,lookback=lookback)
test_x , test_y = shift(test_set,lookback=lookback)


train_x = np.reshape(train_x, (train_x.shape[0],1 ,train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0],1 ,test_x.shape[1]))

# model creation
model = Sequential()
model.add(LSTM(4, input_dim=lookback))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
# import pdb
# pdb.set_trace()
print 'training'
model.fit(train_x,train_y,batch_size=2,verbose=1,nb_epoch=100)
test_score = model.evaluate(test_x,test_y,verbose=1 )

print '\n'
print 'test score is {}'.format(test_score)
print 'testscore sqrt  is {}'.format(math.sqrt(test_score))
train_score = model.evaluate(train_x,train_y,verbose=1)
print 'train score is {}'.format(train_score)
print 'train score sqrt  is {}'.format(math.sqrt(train_score))

train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

train_predict =  np.squeeze(train_predict)
test_predict =  np.squeeze(test_predict)

train_predict= scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])

train_y = np.squeeze(train_y)
test_y = np.squeeze(test_y)

print 'Train Score: %.2f RMSE' % math.sqrt(mean_squared_error(train_predict, train_y ) )
print 'Test Score: %.2f RMSE' % math.sqrt(mean_squared_error(test_predict, test_y ) )