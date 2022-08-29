#%%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.fftpack
# And the tf and keras framework, thanks to Google
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
#%% DATA (time series)
#%% creating time series data to mimic real signal
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

y_test = y_train + np.random.normal(0,10**(-SNR/20),len(y_train))# noisy signal
#y_test = np.sin(X_test)
#y_test  = np.sin(X_train) + np.sin(3*X_train)/3 + np.sin(5*X_train)/5 # multiple harmonics
#y_test = y =np.sin(0.6*X_train+0.5)+ np.sin(np.pi/2) + np.sin(90*0.6*X_train+0.5)# + np.sin(tm*(-3*np.pi/2))# + 0.01*np.random.randn(len(tm)) # dont use

y2=np.sin(3*2*X_train)/3
y3=np.sin(5*2*X_train)/5
y4=np.sin(7*2*X_train)/7
y5=np.sin(9*2*X_train)/9

#y_test=y_train+y2+y3+y4+y5 # distorted multiple harmonics 
n_features = 1

train_series = y_train.reshape((len(y_train), n_features))
test_series  = y_test.reshape((len(y_test), n_features))
#%% Plots for test and train data
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
#%% Plotting FFT of test data
# 64 point FFT
N = 64
yf = np.fft.fft(y_test[0:N])
xf = np.linspace(0.0, 1.0/(2*timestep), int(N/2))
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]), linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title("FFT of test signal", fontsize=16)
plt.legend()
plt.show()
#%% NN model
def fft_model():
  model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu,
                        input_shape=(train_data.shape[1],)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128)
  ])
  model.summary()
  return model
#%% Training the NN 
Neu = 64
num_train_batch = 1
num_batches = 10000
train_data = np.random.normal(0,1,(num_batches, Neu*2))
train_labels = np.random.normal(0,1,(num_batches, Neu*2))
model = fft_model()
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()]) #metrics=['accuracy'])

for k in range(num_train_batch):
  for el in range(num_batches):
    fft_in = train_data[el,0::2] + 1j*train_data[el,1::2]
    train_labels[el,0::2]=scipy.fftpack.fft(fft_in).real
    train_labels[el,1::2]=scipy.fftpack.fft(fft_in).imag
  EPOCHS = 100
  strt_time = datetime.datetime.now()
  history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=2)
                  #  callbacks=[])
  curr_time = datetime.datetime.now()
  timedelta = curr_time - strt_time
  dnn_train_time = timedelta.total_seconds()
  print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
  plt.plot(history.epoch, np.array(history.history['val_loss']),
            label = 'Val loss')
  plt.show()
  train_data = np.random.normal(0,1,(num_batches, Neu*2))
#%% Plotting FFT Prediction 
fft_in = np.zeros((1,2*Neu))
fft_in[:,0::2]=y_test[0:Neu]
fft_out = model.predict(fft_in).flatten()
fft_out = fft_out[0::2] + 1j*fft_out[1::2]
plt.plot(xf, 2.0/Neu * np.abs(fft_out[0:Neu//2]),'c', label='Predicted')
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]),'r', label='Original')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title("FFT prediction", fontsize=16)
plt.legend(loc="upper right")
plt.show()
