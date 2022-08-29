import numpy as np
import matplotlib.pyplot as plt
import datetime
#import scipy.fftpack
#from scipy.fftpack import fft 
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# %%circuit implementation

  # import pandas as pd
  # dataset = pd.read_csv('circuit_waveform.csv')
  # x = dataset.drop(columns=['except_column']) # columns that we are drooping
  # y = dataset['except_column(1=m, 0=b)'] # y and x column set
  # from sklearn.model_selection import train_test_split
  # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  # # model
  # model = tf.keras.models.Sequential()
  # model.add(tf.keras,layers.LSTM(64, input_shape=X_train.shape, activation='relu'))
  # model.add(tf.kekras.layers.Dense(32, activation='relu'))
  # model.add(tf.kekras.layers.Dense(1, activation='relu'))
  # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

  # filepath= "C:\\Users\\Habsman\\Desktop\\codes\\recursive\\saved_models_2"
  # cp1 = ModelCheckpoint(filepath, save_best_only=True)

  # model.fit(X_train, y_train, epochs=100, callbacks=[cp1]) #history.model.fit
  # #verbose=0)
  # model.evaluate(X_test, y_test) # evaluate algorithm, testing set on the y test, comparing what model think what y test should be compared to what it actually is
  # model.load_weights("C:\\Users\\Habsman\\Desktop\\codes\\recursive\\saved_models_2")



# %% working model 
# Time series prediction model

def dnn_keras_tspred_model():
  model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.relu,
                        input_shape=(train_data.shape[1],)),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  model.summary()
  return model


# %% Data stream

#creating time series, real signal mimic
num_train_data = 4000
num_test_data = 1000
timestep = 0.1
look_back = 20
X_train = np.arange(0, (num_train_data+num_test_data)*timestep, timestep);
#X_train = np.arange(0, 100, 0.5)
y_train = np.sin(X_train)

SNR = 10
X_test = np.arange((num_train_data+num_test_data)*timestep, (num_train_data+num_test_data)/5, timestep);
#X_test = np.arange(100,200,0.5) 

#y_test = y_train + np.random.normal(0,10**(-SNR/20),len(y_train)) # Noisy signal
#y_test = np.sin(X_test)
#y_test  = np.sin(X_train) + np.sin(3*X_train)/3 + np.sin(5*X_train)/5 # multiple harmonics 

y2=np.sin(3*2*X_train)/3
y3=np.sin(5*2*X_train)/5
y4=np.sin(7*2*X_train)/7
y5=np.sin(9*2*X_train)/9

y_test=y_train+y2+y3+y4+y5 # Distorted multiple harmonics
n_features = 1

train_series = y_train.reshape((len(y_train), n_features))
test_series  = y_test.reshape((len(y_test), n_features))

# %% plots# plot the function / noise on signal 
plt.plot(X_train[0:100],y_train[0:100], 'k', label='train data')
plt.plot(X_train[0:100],y_test[0:100], 'r', label='test/input data') # red one is the noisy signal/ multiple harmonics
plt.title('Train and Test data')
plt.xlabel('Time')
plt.ylabel('Function')
plt.legend(loc="lower left")
plt.savefig('function.png')
plt.show()

plt.plot(X_train[0:300],y_train[0:300], 'k', label='train data')
plt.plot(X_train[300:500],y_test[300:500], 'r', label='test data') # red one is the noisy signal/ multiple harmonics
plt.title('Training data')
plt.xlabel('Time')
plt.ylabel('Function')
plt.legend(loc="lower left")
plt.savefig('function.png')
plt.show()

plt.plot(X_train,y_test, color='c', label='Noisy')
plt.plot(X_train,y_train,  color='k', label='Clean')
plt.legend()

#%% perform fft

yf = np.fft.fft(y_train)
yf2 = np.fft.fft(y_test)
#xf = np.fft.fftfreq(len(X_train))
n = len(X_train)


# fhat = np.fft.fft(y_test,n)
# PSD = fhat * np.conj(fhat) / n # power spectrun
freq = (1/(timestep*n)) * np.arange(n)
L = np.arange(1,np.floor(n/2),dtype= 'int')


plt.plot(freq, np.abs(yf2), linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title("Frequency domain of the signal", fontsize=16)
plt.legend()
plt.show()

fig,axs = plt.subplots(2,1)
plt.sca(axs[0])
plt.plot(X_train,y_test, color='c', label='Noisy')
plt.plot(X_train,y_train,  color='k', label='Clean')
#plt.xlim(X_train[0],X_train[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], np.abs(yf2)[L])# color='c', Label='Noisy')
plt.xlim(freq[L[0]],freq[L[-1]])
plt.legend()
plt.show()

#%% (primary example) prepare the train_data and train_labels 
dnn_numinputs = 64
num_train_batch = 0
train_data = []
for k in range(num_train_data-dnn_numinputs-1):
  train_data = np.concatenate((train_data,y_test[k:k+dnn_numinputs]));
  num_train_batch = num_train_batch + 1  
train_data = np.reshape(train_data, (num_train_batch,dnn_numinputs))
train_labels = y_train[dnn_numinputs:num_train_batch+dnn_numinputs]

print(y_train.shape, train_data.shape, train_labels.shape)

# %% first model 
model = dnn_keras_tspred_model()
filepath= "C:\\Users\\Habsman\\Desktop\\codes\\recursive\\saved_models_MLP"
cp = ModelCheckpoint(filepath, save_best_only=True)
# training defining epochs

model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()]) #metrics=['accuracy'])
EPOCHS = 100
strt_time = datetime.datetime.now()
history = model.fit(train_data, train_labels, epochs=EPOCHS, 
                  validation_split=0.2, #verbose=0,
                  callbacks=[cp])
curr_time = datetime.datetime.now()
timedelta = curr_time - strt_time
dnn_train_time = timedelta.total_seconds()
print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")

# %% validation and training plots 
plt.plot(history.epoch, np.array(history.history['val_loss']),
            label = 'Val loss')

plt.xlabel('epochs')
plt.ylabel('Validation_Loss')
plt.savefig('validation_loss.png')
plt.show()
#plot the training and validation curves
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.figure()
plt.plot(epochs, loss,'b', label='Training loss',linewidth=2)
plt.plot(epochs, val_loss,'r', label='Validation loss',linewidth=2)
plt.title('Training and validation losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('validation_and_loss.png')
plt.show()

# %% Predictions
# test how well DNN predicts now
num_test_batch = 0
strt_idx = num_train_batch
test_data=[]
for k in range(strt_idx, strt_idx+num_test_data-dnn_numinputs-1):
  test_data = np.concatenate((test_data,y_test[k:k+dnn_numinputs]));
  num_test_batch = num_test_batch + 1  
test_data = np.reshape(test_data, (num_test_batch, dnn_numinputs))
test_labels = y_train[strt_idx+dnn_numinputs:strt_idx+num_test_batch+dnn_numinputs]


dnn_predictions = model.predict(test_data).flatten()
keras_dnn_err = test_labels - dnn_predictions

# %% final plots 

plt.plot(dnn_predictions[0:100],label='Predicted')
plt.plot(test_labels[0:100], color='r',label='Desired') #OG data
plt.title('Prediction vs Desired results')
plt.xlabel('Time')
plt.ylabel('Function Value')
plt.savefig('prediction.png')
plt.legend(loc="lower left")
plt.show()

plt.plot(X_train[0:250],y_train[0:250], 'k', label='train data')
plt.plot(X_train[250:400],y_test[250:400], 'r', label='test data') # red one is the noisy signal/ multiple harmonics
plt.plot(X_train[400:500],dnn_predictions[400:500], c='b', linestyle = ':', label='prediction')
plt.title('Prediction summary')
plt.xlabel('Time')
plt.ylabel('Function')
plt.legend(loc="lower left")
plt.savefig('function.png')
plt.show()


yff = np.fft.fft(dnn_predictions)
xff = np.fft.fftfreq(len(dnn_predictions))

plt.plot(xff, np.abs(yff), label='prediction data')
plt.title('Output FFT')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend(loc="lower left")
plt.show();




