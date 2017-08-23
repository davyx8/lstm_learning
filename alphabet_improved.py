import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

numpy.random.seed(8)

alphabet='abcdefghijklmnopqrstuvwxyz'
i_to_c = dict((i,c) for i ,c in enumerate(alphabet))
c_to_i = dict((c,i) for i ,c in enumerate(alphabet))
seq_length=3
data_x=[]
def split_to_n(seq,n):
    while seq:
        if len(seq)<=n:
            yield seq[:n],seq[-1]
        else:
            yield seq[:n],seq[n]
        seq = seq[1:]

seq=split_to_n(alphabet,seq_length)
data_x_orig = [next(seq)[0]  for iter in range(len(alphabet)-seq_length) ]
seq=split_to_n(alphabet,seq_length)
data_y = [next(seq)[1]  for iter in range(len(alphabet)-seq_length) ]
convert = lambda cur_seq : map(c_to_i.get,cur_seq)
data_x = map( convert , data_x_orig)
data_y = map(convert  , data_y)
X = numpy.reshape(data_x,(len(data_x) , seq_length,1))
X = X/ float(len(alphabet))
y = np_utils.to_categorical(data_y)
print X.shape
print y.shape

model= Sequential()
model.add(LSTM(16, batch_input_shape=(1,X.shape[1],X.shape[2]) ,stateful=True))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
for i in range(300):
   model.fit(X,y,epochs=1,batch_size=1,verbose=2,shuffle=False)
   model.reset_states()

scores=model.evaluate(X,y,batch_size=1,verbose=1)
print scores[1]*100
i=0
model.reset_states()
import pdb ; pdb.set_trace()
for letter in X:
    
    print data_x_orig[i]+'->' + i_to_c[ numpy.argmax( model.predict(numpy.reshape(letter,(1,len(letter),1))))]
    i+=1
