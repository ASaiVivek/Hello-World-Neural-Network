import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, optimizers
import numpy as np


model=keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])#units=dimensionality of the output space and input_shape is 1
model.compile(optimizer='sgd',loss='mean_squared_error')#Stochastic gradient descent Optimizer

#train_data
xs=np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)#features
ys=np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)#labels

#test_data
xt=np.array([-1.0,0.0,1.0],dtype=float)
yt=np.array([-3.0,-1.0,1.0],dtype=float)

model.fit(xs,ys,epochs=500)

#predicting values
print(model.predict([10.0]))
print(model.predict([100.0]))

#evaluating model based on test_data
loss=model.evaluate(xt,yt,verbose=1) #verbose =0,1 or 2:for representing epochs
print('\n Loss={0:.2f}%'.format(loss*100.0))
print('\n Accuracy={0:.2f}%'.format((1-loss)*100.0))