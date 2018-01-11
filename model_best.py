from __future__ import print_function
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Bidirectional
from keras.layers import LSTM,Masking,GRU
from keras.optimizers import adam
import numpy as np
import math

def labelpadding(len,cat,sil=39):
    zero = np.zeros([len,cat])
    #zero[:,sil] = 1
    #print(cat,sil,zero.shape)
    return zero

def reshape_xydata(dictx_data, dicty_data, maxlen):
    overlap = int(maxlen/2)
    wavname = list(dictx_data.keys()) # wavname list
    wavnum = len(wavname)
    dim = dictx_data[wavname[0]].shape[1] # input dimensions
    cat = dicty_data[wavname[0]].shape[1] # numbers of phones
    ### Transform x_data into shape (frames, timesteps, dim)
    ### Transform y_data into shape (frames, cat)
    trframe_num = sum([int(math.ceil(dictx_data[name].shape[0]*1./overlap))-1 for name in wavname[0:3100]])
    valframe_num = sum([int(math.ceil(dictx_data[name].shape[0]*1./overlap))-1 for name in wavname[3100:]])
    x_train = np.zeros([trframe_num, maxlen, dim])
    x_valid = np.zeros([valframe_num, maxlen, dim])
    y_train = np.zeros([trframe_num, maxlen,cat])
    y_valid = np.zeros([valframe_num, maxlen,cat])
    print('phone catagories:',cat)
    trainframe = 0
    validframe = 0
    for i in range(wavnum):
        wavdata = np.float64(dictx_data[wavname[i]])
        wavdata = (wavdata-wavdata.mean(axis = 0))/wavdata.std(axis = 0)
        frlen = wavdata.shape[0]
        wavlabel = dicty_data[wavname[i]]
        padlen = int(math.ceil(frlen*1./overlap))*overlap-frlen
        frame = int(math.ceil(frlen*1./overlap))-1

        wavdata = np.row_stack((wavdata,np.zeros([padlen,dim])))
        wavlabel = np.row_stack((wavlabel,labelpadding(padlen,cat)))
        if i < 3100: # training
            for j in range(frame):
                start = j*overlap
                end = start+maxlen
                x_train[trainframe+j] = wavdata[start:end]
                y_train[trainframe+j] = wavlabel[start:end]
            trainframe = trainframe+frame
        else: # validation
            for j in range(frame):
                start = j*overlap
                end = start+maxlen
                x_valid[validframe+j] = wavdata[start:end]
                y_valid[validframe+j] = wavlabel[start:end]
            validframe = validframe+frame
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    print('x_valid shape:', x_valid.shape)
    print('y_valid shape:', y_valid.shape)


    return x_train, y_train, x_valid, y_valid, dim, cat


def LSTMmodel(dim, cat):
    print('Build model...')
    model = Sequential()
    model.add(Masking(mask_value=0., batch_input_shape=(None,None,dim)))
    model.add(Bidirectional(GRU(units=256, activation='tanh',dropout = 0.2,recurrent_dropout=0.2,return_sequences=True)))
    #model.add(LSTM(units=256, activation='tanh',return_sequences=True)), dropout = 0.1
    model.add(GRU(units=cat,  activation='softmax',return_sequences=True))
    #model.add(Dense(cat,activation='softmax'))
    print(model.summary())
    return model
batch_size = 250
#look_back = 3
maxlen = 100
if __name__ == '__main__':
    print('Loading data...')
    ### Load data
    dictx_data = np.load('best_x_data.npy').item()
    dicty_data = np.load('best_y_data.npy').item()
    x_train, y_train, x_valid, y_valid, dim, cat = reshape_xydata(dictx_data, dicty_data, maxlen)
    print(dim,cat)
    #x_train = x_train[0:900000]
    #y_train = y_train[0:900000]
    #x_valid = x_valid[0:210000]
    #y_valid = y_valid[0:210000]
    model = LSTMmodel(dim,cat)
    # Run training
    print('Train...')
    adam = adam(lr = 0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    checkpointer = ModelCheckpoint(filepath='best100_rnn_model_cp002.h5', verbose=0, save_best_only=True,monitor='val_acc')
    model.fit(x_train, y_train,
              callbacks=[checkpointer],
              verbose=1,
              shuffle=True,
              batch_size=batch_size,
              epochs=100,
              validation_data=(x_valid, y_valid))

    model.save('best_rnn_models.h5')