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
encoded = [vocab_to_int[c] for c in text]
encoded=keras.utils.to_categorical(encoded)

encoded=np.asarray(encoded)


def createDataSet(D,look):
    X,y=[],[]
    for i in range(0,len(D)-look-1):
        X.append(D[i:i+look])
        y.append(D[i+look])
        
    return np.asarray(X),np.asarray(y)    



look=7
X,y=createDataSet(encoded[:len(encoded)//100],look)
s=X[len(X)-1300].tolist()








model=keras.models.load_model('checkpoints/weights.33-0.75.hdf5')
i='a'
#while i!=' ':
for z in range(200):
    l=s.copy()
    h=model.predict_classes(np.asarray(l).reshape(1,look,74))[0]
    i=int_to_vocab[np.argmax(s.pop(0))]
    print(i,end='')
    s.append(inttovector(h))
