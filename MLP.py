
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from numpy import shape

visible = Input(shape(2, ))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)
model.summary()

