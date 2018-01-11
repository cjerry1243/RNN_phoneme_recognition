# This script reads the mfcc data with dimension = 39,
# and transforms it to frame-wise sequence data for training.
# Besides, labels will be read in this script and condensed into 39 phones.
# We will split the data into 2 sets(taining and validation sets) randomly,
# and then x_train, y_train, x_valid, y_valid will be later used for training process.
#
# dict48_39 is the mapping dictionary from 48 to 39 phones
# phone_39 orders 39 phones
# dictx_data is the input data (dim=39) dictionary with the wavname as its key
# dictx_data is the label (dim=39) dictionary with the wavname as its key

import numpy as np
import pandas as pd

def find_all(str,char):
    beg = 0
    index = 1
    allcharindex = []
    while(index>-1):
        index = str.find(char,beg,len(str))
        if index == -1:
            break
        allcharindex.append(index)
        beg = index+1
    return allcharindex



def make_input_dict(path):
    pdx = pd.read_csv(path,header=None,sep=' ')
    pdx = pdx.values
    dictx_data = {}
    eow = 0
    for i,name in enumerate(pdx[:,0]):
        name = name[0:find_all(name,'_')[1]]
        if i ==0:
            wavname = name
        elif wavname != name:
            dictx_data[wavname] = pdx[eow:i,1:]
            wavname = name
            eow = i
        elif i == pdx.shape[0]-1:
            dictx_data[wavname] = pdx[eow:i+1, 1:]
    return dictx_data

def phoneslist(path):
    phones_39 = []
    fmap = open(path,'r')
    dict48_39 = {}
    for lines in fmap.readlines():
        lines = lines.strip().split()
        if lines[1] not in phones_39:
            phones_39.append(lines[1])
        dict48_39[lines[0]] = lines[1]
    fmap.close()

    return phones_39, dict48_39


def makelabelarray(dim,index):
    labelarray = np.zeros([1,dim+1])
    labelarray[0,index] = 1
    return labelarray

def make_labels_dict(path,phones_39, dict48_39,dim=39):
    g = open(path+'label/train.lab','r')
    dicty_data = {}
    wavname = ''
    for lines in g.readlines():
        index = phones_39.index(dict48_39[lines[lines.strip().index(',')+1:-1]])
        label = makelabelarray(dim,index)
        if wavname != lines[0:find_all(lines,'_')[1]]:
            wavname = lines[0:find_all(lines,'_')[1]]
            dicty_data[wavname] = np.empty([0,dim+1])
        dicty_data[wavname] = np.row_stack((dicty_data[wavname],label))
    g.close()
    return dicty_data

if __name__ == '__main__':
    dim = 39
    path = 'mfcc/train.ark'
    dictx_data = make_input_dict(path)
    inputpath = ''
    phones_39, dict48_39 = phoneslist(inputpath)
    dicty_data = make_labels_dict(inputpath,phones_39, dict48_39)
    np.save('x_data.npy',dictx_data)
    np.save('y_data.npy',dicty_data)


