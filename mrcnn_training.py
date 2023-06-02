# Author: Ashok Parmar

import matplotlib

from matplotlib import pyplot as plt
import numpy as np
from numpy import save, load
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import os
import pickle


import h5py


classes = ['bpsk', 'qpsk', '8psk','16qam']
print('Loading dataset....')
with h5py.File("/home/dataset/NomaDL_2022_v2.mat", 'r') as Xd:
	#Xd = f3
	X = np.array(Xd['data_Y'])
	lbl = np.array(Xd['true_Mods'])
	snrs = np.array(Xd['snrs'])
cplx = X['real'] + X['imag']*1j
c_t = cplx.transpose()
Xs = np.zeros((c_t.shape[0],2,c_t.shape[1]))
Xs[:,0,:] = c_t.real
Xs[:,1,:] = c_t.imag

print("preprocessing data")
np.random.seed(2016)
n_examples = Xs.shape[0]
n_train = int(n_examples * 0.6)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = Xs[train_idx]
X_test =  Xs[test_idx]

snrs = list(snrs.reshape(snrs.shape[0]))

trainy = list(lbl[train_idx])
testy = list(lbl[test_idx])

Y_train = to_categorical(trainy,4)
Y_test = to_categorical(testy,4)

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp,X_test.shape)
nb_epoch = 100    # number of epochs to train on

batchSize = 500  # training batch size

def amc(layer_in):
	conv1 = Conv2D(16, (1,8), padding='same', activation='relu')(layer_in)
	#pool1 = MaxPooling2D((1,3))(conv1)
	conv3 = Conv2D(16, (1,1), padding='same', activation='relu')(layer_in)
	canc = Concatenate()([conv1, conv3])
	return canc


def model7(in_shape):
	dr = 0.4 # dropout rate (%)
 
	input1 = tf.keras.Input(shape = in_shape, name = 'in1')
	h1 = layers.Reshape((2,in_shape[1],1), input_shape=in_shape)(input1)
	h = layers.Conv2D(64, kernel_size=(2, 8),activation='relu',padding='same')(h1)
	#h = layers.MaxPooling2D((1,8))(h)
	h = layers.Conv2D(32, kernel_size=(2, 8), activation='relu',padding='same')(h)
	h = amc(h)
	h = layers.AveragePooling2D((1,3))(h)
	h = layers.Conv2D(16, kernel_size=(1, 3),activation='relu',padding='same')(h)
	h = amc(h)
	h = layers.AveragePooling2D((1,8))(h)
	h = layers.Conv2D(8, kernel_size=(1, 3),activation='relu',padding='same')(h)
	h = layers.Flatten()(h)
	h = layers.Dense(128, activation='relu')(h)
	h = layers.Dense(24, activation='relu')(h)
	output = layers.Dense(4, activation='softmax')(h)
	model = tf.keras.Model(input1,output)
	return model

model = model7(in_shp)

opt = keras.optimizers.Adam(0.001)
model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()  
import os


radio_train = model.fit(X_train, Y_train, batch_size=batchSize,epochs=nb_epoch,verbose=1,validation_data=(X_test, Y_test))

history = radio_train.history

score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batchSize)
print(score)


test_Y_hat = model.predict(X_test, batch_size=batchSize)
plt.rcParams["figure.figsize"] = (10,10)
predicted_classes1 = np.argmax(np.round(test_Y_hat), axis=1)
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
confnorm = confusion_matrix(testy, predicted_classes1)

ConfusionMatrixDisplay.from_predictions(y_true=testy,y_pred=predicted_classes1,xticks_rotation=45, display_labels =classes, normalize = 'true',cmap='Blues', values_format =".2f")


test_snrs = np.array(list(map(lambda x: snrs[x], test_idx)))
snr_list= np.unique(test_snrs)

def snr_based_performance(snr, test_SNRs, test_idx, X_test1, Y_test, model,model_name):
	# extract classes @ SNR
	test_X_i = X_test1[np.where(np.array(test_SNRs)==snr)]
	test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    
	# estimate classes
	test_Y_i_hat = model.predict(test_X_i)
	predicted_classes2 = np.argmax(np.round(test_Y_i_hat), axis=1)
	test_class = np.argmax(np.round(test_Y_i), axis=1)
	conf = np.zeros([len(classes),len(classes)])
	confnorm = np.zeros([len(classes),len(classes)])
	for i in range(0,test_X_i.shape[0]):
		j = list(test_Y_i[i,:]).index(1)
		k = int(np.argmax(test_Y_i_hat[i,:]))
		conf[j,k] = conf[j,k] + 1
	
	for i in range(0,len(classes)):
		confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
	
	plt.figure()
	
	
	cm = ConfusionMatrixDisplay.from_predictions(test_class,predicted_classes2,xticks_rotation=45,
	                                            display_labels =classes, normalize = 'true',cmap='Blues',
	                                            values_format =".2f")

	plt.close()
	cor = np.sum(np.diag(conf))
	ncor = np.sum(conf) - cor
	print("Accuracy at", snr,": ", cor / (cor+ncor))
	acc = 1.0*cor/(cor+ncor)
	return acc

acc = {}
for snr in snr_list:
    acc[snr]  = snr_based_performance(snr, test_snrs, test_idx, X_test, Y_test, model,model_name)


try:
    acc_file = open('/home/Results/acc.pkl', 'wb')
    pickle.dump(acc, acc_file)
    acc_file.close()
except:
    print("Something went wrong")
    
# Plot accuracy curve
plt.plot(snr_list, list(map(lambda x: acc[x], snr_list)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
    

