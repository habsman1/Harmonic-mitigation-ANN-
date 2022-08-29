#%% libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint
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

#y_test = y_train + np.random.normal(0,10**(-SNR/20),len(y_train))# noisy signal
#y_test = np.sin(X_test)
#y_test  = np.sin(X_train) + np.sin(3*X_train)/3 + np.sin(5*X_train)/5 # multiple harmonics
#y_test = y =np.sin(0.6*X_train+0.5)+ np.sin(np.pi/2) + np.sin(90*0.6*X_train+0.5)# + np.sin(tm*(-3*np.pi/2))# + 0.01*np.random.randn(len(tm)) # dont use

y2=np.sin(3*2*X_train)/3
y3=np.sin(5*2*X_train)/5
y4=np.sin(7*2*X_train)/7
y5=np.sin(9*2*X_train)/9

y_test=y_train+y2+y3+y4+y5 # distorted multiple harmonics 
n_features = 1

train_series = y_train.reshape((len(y_train), n_features))
test_series  = y_test.reshape((len(y_test), n_features))
#%% plots the time series signal / noise on signal 
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
#%% FFT plots 
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
plt.title("Frequency domain of noisy signal", fontsize=16)
plt.legend()
plt.show()

plt.plot(freq, np.abs(yf), linestyle='-', color='blue')
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title("Frequency domain of desired signal", fontsize=16)
plt.legend()
plt.show()

fig,axs = plt.subplots(2,1)
plt.sca(axs[0])
plt.plot(X_train,y_test, color='c', label='Noisy')
plt.plot(X_train,y_train,  color='k', label='Clean')
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], np.abs(yf2)[L])# color='c', Label='Noisy')
plt.xlim(freq[L[0]],freq[L[-1]])
plt.legend()
plt.show()
#%% preparing data to be trained and train labels and reshaping from 2D to 3D

dnn_numinputs = 64
num_train_batch = 0
train_data = []
for k in range(num_train_data-dnn_numinputs-1):
  train_data = np.concatenate((train_data,y_test[k:k+dnn_numinputs]));
  num_train_batch = num_train_batch + 1  
train_data = np.reshape(train_data, (num_train_batch,dnn_numinputs))
train_labels = y_train[dnn_numinputs:num_train_batch+dnn_numinputs]
train_data = train_data.reshape(train_data.shape[0], 64, 1)
print(y_train.shape, train_data.shape, train_labels.shape)
#%% Training RNN model
timesteps = 64
n_features = 1
model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(timesteps,n_features), return_sequences=True))
model.add(LSTM(32, activation='tanh', return_sequences=True))
model.add(LSTM(8, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
filepath= "C:\\Users\\Habsman\\Desktop\\codes\\recursive\\saved_models_RNN"
cp = ModelCheckpoint(filepath, save_best_only=True)
# training defining epochs
#model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()]) #metrics=['accuracy'])
model.summary()

EPOCHS = 100
strt_time = datetime.datetime.now()
history = model.fit(train_data, train_labels, epochs=EPOCHS, 
                  validation_split=0.2)#, #verbose=0,
                  #callbacks=[cp])
curr_time = datetime.datetime.now()
timedelta = curr_time - strt_time
dnn_train_time = timedelta.total_seconds()
print("DNN training done. Time elapsed: ", timedelta.total_seconds(), "s")
#%% plotting val and training 
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
#%% Testing how well the Designed model predicts
num_test_batch = 0
strt_idx = num_train_batch
test_data=[]
for k in range(strt_idx, strt_idx+num_test_data-dnn_numinputs-1):
  test_data = np.concatenate((test_data,y_test[k:k+dnn_numinputs]));
  num_test_batch = num_test_batch + 1  
test_data = np.reshape(test_data, (num_test_batch, dnn_numinputs))
test_labels = y_train[strt_idx+dnn_numinputs:strt_idx+num_test_batch+dnn_numinputs]
test_data = test_data.reshape(test_data.shape[0], 64, 1)

dnn_predictions = model.predict(test_data)#.flatten()
dnn_predictions = dnn_predictions[:, 0, 0]

keras_dnn_err = test_labels - dnn_predictions

#%% plotting prediction summary and output FFT
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

