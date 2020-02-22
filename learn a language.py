import ctypes
from sklearn import model_selection
import numpy as np
import pandas as pd
import random
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
encoded = [vocab_to_int[c] for c in text]
encoded=keras.utils.to_categorical(encoded)
print(encoded.shape)
encoded=np.asarray(encoded)


'''
for preprocessing
with open('anna.txt', 'w') as f:
    for i in X:
        f.write(i.replace('\n',' ')+'\n')
'''

def createDataSet(D,look):
    X,y=[],[]
    for i in range(0,len(D)-look-1,1):
        X.append(D[i:i+look])
        y.append(D[i+look])
        
    return np.asarray(X),np.asarray(y)       

def inttovector(n):
    a=[0. for i in range(74)]
    a[n]=1
    return a

look=12
X,y=createDataSet(encoded[:len(encoded)//50],look)
print(X.shape)
X=X.reshape(X.shape[0],look,74)
print(X.shape,y.shape)

es=keras.callbacks.EarlyStopping(monitor='categorical_accuracy',patience=10)
save=keras.callbacks.ModelCheckpoint('checkpoints/weights.{epoch:02d}-{categorical_accuracy:.2f}.hdf5',monitor='categorical_accuracy',period=10)
hidsize=2500
model=keras.Sequential()
#model.add(keras.layers.Input((look,74)))
#model.add(keras.layers.LSTM(hidsize,return_sequences=True,activation='sigmoid'))
model.add(keras.layers.LSTM(hidsize,activation='sigmoid'))
#model.add(keras.layers.Dropout(0.4))
#model.add(keras.layers.Dense(30,'relu'))
model.add(keras.layers.Dense(74,'softmax'))
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])

ep=0


def on_epoch_end(epoch, _):
    global ep
    ep+=1
    if ep%5!=0:
        return
    print('*************************************************************************')
    r=random.randrange(len(X)-look-1)
    s=encoded[r:r+look].tolist()
    for z in range(160):    
        l=s.copy()
        h=model.predict_classes(np.asarray(l).reshape(1,look,74))[0]
        i=int_to_vocab[np.argmax(s.pop(0))]
        print(i,end='')
        s.append(inttovector(h))
    print('*************************************************************************')
end=keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)


model.fit(X,y ,epochs=5000,callbacks=[save,end,es],verbose=2,use_multiprocessing=True,batch_size=256)



