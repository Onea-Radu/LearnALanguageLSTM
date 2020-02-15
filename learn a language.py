import ctypes
from sklearn import model_selection
import numpy as np
import pandas as pd
hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudart64_100.dll")
hllDll1 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cublas64_100.dll")
hllDll2 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudnn64_7.dll")
from tensorflow import keras







with open('anna.txt', 'r') as f:
    text=f.read()
    #X=text.split('\n')
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
encoded=keras.utils.to_categorical(encoded,74,'int32')

'''
for preprocessing
with open('anna.txt', 'w') as f:
    for i in X:
        f.write(i.replace('\n',' ')+'\n')
'''

def createDataSet(D,look):
    X,y=[],[]
    for i in range(len(D)-look-1):
        X.append(D[i:(i+look)])
        y.append(D[i+look])
        
    return np.array(X),np.array(y)       



look=9
X,y=createDataSet(encoded,look)

X=X[0:len(X)//10]
y=y[0:len(y)//10]
print(X.shape,y.shape)

save=keras.callbacks.ModelCheckpoint('checkpoints/weights.{epoch:02d}-{categorical_accuracy:.2f}.hdf5',monitor='categorical_accuracy',period=5)
hidsize=74
model=keras.Sequential()
model.add(keras.layers.Input((look,74,)))
model.add(keras.layers.LSTM(hidsize,return_sequences=True,activation='sigmoid'))
model.add(keras.layers.LSTM(hidsize,activation='sigmoid'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(700,'relu'))
model.add(keras.layers.Dense(700,'relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(700,'relu'))
model.add(keras.layers.Dense(700,'relu'))
model.add(keras.layers.Dense(74,'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()
model.fit(X,y,epochs=501,callbacks=[save],verbose=2)



