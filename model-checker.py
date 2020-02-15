import ctypes
from sklearn import model_selection
import numpy as np
import pandas as pd
#hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudart64_100.dll")
#hllDll1 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cublas64_100.dll")
#hllDll2 = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\cudnn64_7.dll")
from tensorflow import keras

def inttovector(n):
    a=[0. for i in range(74)]
    a[n]=1
    return a
    





with open('anna.txt', 'r') as f:
    text=f.read()
    #X=text.split('\n')
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
encoded=keras.utils.to_categorical(encoded,74,'int32')

def createDataSet(D,look):
    X,y=[],[]
    for i in range(len(D)-look-1):
        X.append(D[i:(i+look)])
        y.append(D[i+look])
        
    return np.array(X),np.array(y)       

look=9
X,y=createDataSet(encoded,look)
s=X[len(X)//302].tolist()
l=s.copy()
z=(np.array([l,])).astype('float32')





model=keras.models.load_model('checkpoints/weights.06-0.17.hdf5')

for i in range(120):
    h=model.predict_classes(z)[0]
    print(int_to_vocab[np.argmax(z[0][0])],end='')
    s.pop(0)
    s.append(inttovector(h))
    l=s.copy()
    z=(np.array([l,])).astype('float32')

    