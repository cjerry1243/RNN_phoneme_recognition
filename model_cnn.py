from __future__ import print_function
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.layers import Dense,Flatten,Masking
from keras.layers import LSTM,Conv2D,TimeDistributed,Bidirectional
from keras.optimizers import adam
import numpy as np

def labelpadding(len,cat,non=39):
    zero = np.zeros([len,cat])
    zero[:,non] = 1
    return zero

def reshape_xydata(dictx_data, dicty_data, maxlen):
    wavname = list(dictx_data.keys()) # wavname list
    wavnum = len(wavname)
    dim = dictx_data[wavname[0]].shape[1] # input dimensions
    cat = dicty_data[wavname[0]].shape[1] # numbers of phones

    ### Transform x_data into shape (frames, timesteps, dim)
    ### Transform y_data into shape (frames, cat)
    #trframe_num = sum([dictx_data[name].shape[0] for name in wavname[0:3000]])
    #valframe_num = sum([dictx_data[name].shape[0] for name in wavname[3000:]])
    x_train = np.zeros([3000, maxlen, dim])
    x_valid = np.zeros([wavnum-3000, maxlen, dim])
    y_train = np.zeros([3000,maxlen,cat])
    y_valid = np.zeros([wavnum-3000,maxlen,cat])

    #trainframe = 0
    #validframe = 0
    for i in range(wavnum):
        wavdata = np.float64(dictx_data[wavname[i]])
        wavdata = (wavdata-wavdata.mean(axis = 0))/wavdata.std(axis = 0)
        frlen = wavdata.shape[0]
        wavlabel = dicty_data[wavname[i]]
        padlen = maxlen-frlen
        pre_padlen = int(padlen/2)
        post_padlen = padlen-pre_padlen
        if i < 3000: # training
            x_train[i] = np.row_stack((wavdata,np.zeros([padlen,dim])))
            y_train[i] = np.row_stack((wavlabel,labelpadding(padlen,cat)))
        else: # validation
            x_valid[i-3000] = np.row_stack((wavdata,np.zeros([padlen,dim])))
            y_valid[i-3000] = np.row_stack((wavlabel,labelpadding(padlen,cat)))
    print('x_train shape:', x_train.shape)
    print('x_valid shape:', x_valid.shape)

    return x_train, y_train, x_valid, y_valid, dim, cat


def LSTMmodel(dim, cat):
    print('Build model...')
    model = Sequential()
    model.add(Conv2D(5,[3,1],padding='same',input_shape = (None,dim,1)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(units=256, activation='tanh',return_sequences=True)))#, batch_input_shape=(None,None,dim)))
    model.add(LSTM(units=cat,  activation='softmax',return_sequences=True))
    #model.add(Dense(cat,activation='softmax'))

    return model
batch_size = 50
look_back = 3
maxlen = 777
if __name__ == '__main__':
    print('Loading data...')
    ### Load data
    dictx_data = np.load('x_data.npy').item()
    dicty_data = np.load('y_data.npy').item()
    x_train, y_train, x_valid, y_valid, dim, cat = reshape_xydata(dictx_data, dicty_data, maxlen)

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_valid = x_valid.reshape(x_valid.shape[0],x_valid.shape[1],x_valid.shape[2],1)

    model = LSTMmodel(dim, cat)
    # Run training
    print('Train...')
    adam = adam(lr = 0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    checkpointer = ModelCheckpoint(filepath='Crnn_model_cp.h5', verbose=0, save_best_only=True,monitor='val_acc')
    model.fit(x_train, y_train,
              callbacks=[checkpointer],
              verbose=1,
              shuffle=True,
              batch_size=batch_size,
              epochs=80,
              validation_data=(x_valid, y_valid))

    model.save('Crnn_models.h5')