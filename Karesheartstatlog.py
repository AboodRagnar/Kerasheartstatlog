import keras
from keras.layers import Activation
from keras.models import Sequential
from  keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


GetData=pd.read_csv('T1/heartstatlog.csv')

GetData=GetData.replace("?",-9)

GetData=GetData.astype('float')
GetData=shuffle(GetData)




# Set date as lables and features
X=GetData.drop(columns='presence_of _heart _disease')
Y=GetData['presence_of _heart _disease']



Model=Sequential([

        Dense(26,input_shape=(13,), activation='relu'), 
        Dense(52,activation='relu'),
        Dense(5,activation='softmax')



])


Model.compile(Adam(lr=.001),loss="sparse_categorical_crossentropy",metrics=['accuracy'])


Model.fit(X,Y,validation_split=0.1,batch_size=50,epochs=100,verbose=2,shuffle=True)

print(Model.predict(X,batch_size=50,verbose=0))
