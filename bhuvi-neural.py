import os
import sys
import requests
import pandas

import numpy as np
from keras.layers.core import Activation,Dropout
from keras.models import Sequential
from keras.layers import Dense,normalization
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split	


def stock_prediction(filename):
    print "File ::: " + filename
    FILE_NAME = "tempdataset/"+filename+".csv"
    print "into stock prediction"


    dataset = np.loadtxt(FILE_NAME,delimiter=',')

    train,test=train_test_split(dataset,test_size=0.3)

  #  trainX = dataset[:,0:4]
  #  trainY = dataset[:,3]

  	trainX = train[:,0:4]
    trainY = train[:,3]

    testX = test[:,0:4]
    testY = test[:,3]


    print trainX
    print trainY


    model = Sequential()
    model.add(Dense(len(train), input_dim=4,init='uniform', activation='relu'))
    #model.add(normalization.BatchNormalization())
    model.add(Dense(40))
    #model.add(normalization.BatchNormalization())
    model.add(Dense(40))
    #model.add(normalization.BatchNormalization())
    model.add(Dense(20))
    #model.add(normalization.BatchNormalization())
    # model.add(Dense(16))
    #
    # model.add(Dense(16))
    model.add(Dense(1,activation='relu'))
    #model.add(normalization.BatchNormalization())
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    #model.fit(trainX, trainY, nb_epoch=200,batch_size=128, verbose=2)

    model.fit(trainX, trainY, nb_epoch=200, verbose=2)


    #prediction = model.predict(trainX)
    prediction = model.predict(testX)
    evaluation = model.evaluate(trainX,trainY)
    model.save('model.hdf5')
    model.summary()
    print "Evaluation ::: "
    print evaluation
    print "Accuracy ::: "
    print evaluation*100
    print len(prediction)
    print prediction
    dataset = np.append(dataset,[prediction[0][0]])
    print dataset
    #dataset.append(prediction[0][0].astype(float))

    #SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #plot(model, to_file='model.png')


   # return prediction[0][0].astype(float)
   return prediction

def evaluate(file):
    print "into evaluate"
    print stock_prediction(file)
    # for i in range(7):
    #     price = stock_prediction()
    #     close_list.append(price)
    # print close_list

if __name__=="__main__":
    file = raw_input("Enter File Name").upper()
    evaluate(file)

