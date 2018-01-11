import mfcc
import numpy as np
import sys
from keras.models import load_model

### reshape x_test to input shape (wavnum,maxlen,dim)
def reshape_xdata(dictx_data, maxlen, wavname):
    wavnum = len(wavname)
    dim = dictx_data[wavname[0]].shape[1] # input dimensions

    ### Transform x_data into shape (frames, timesteps, dim)
    x_test = np.zeros([wavnum, maxlen, dim])

    for i in range(wavnum):
        wavdata = np.float64(dictx_data[wavname[i]])
        wavdata = (wavdata-np.mean(wavdata,axis=0))/np.std(wavdata,axis=0)
        frlen = wavdata.shape[0]
        padlen = maxlen-frlen

        x_test[i] = np.row_stack((wavdata,np.zeros([padlen,dim])))

    print('x_test shape:', x_test.shape)
    return x_test

### The order of output wavname
def outputwavname():
    g = open('sample.csv','r')
    wavname = []
    for name in g.readlines()[1:]:
        name = name[0:name.index(',')]
        wavname.append(name)
    g.close()
    return wavname

### trim the phone sequence
def trimming(numlist, silence):
    lastnum = 100
    newlist = []
    for i,num in enumerate(numlist):
        if lastnum == num:
            None
        else:
            newlist.append(num)
            lastnum = num
    if newlist[0]==silence:
        del newlist[0]
    if newlist[-1]==silence:
        del newlist[-1]
    return newlist

### dictionary of 39 phones to 39 output English letters
def letters_39(path):
    phone2letter = {}
    f = open(path+'48phone_char.map','r')
    for lines in f.readlines():
        lines = lines.strip()
        linelist = lines.split('\t')
        phone2letter[linelist[0]]=linelist[2]
    f.close()
    return phone2letter

### dictionary of wav to model output
def dictwav_output(testx_data, wavname, maxlen,model):
    x_test = reshape_xdata(testx_data, maxlen, wavname)
    output = model.predict_classes(x_test)
    wavlength = [testx_data[name].shape[0] for name in wavname]
    dict_out = {}
    start = 0
    end = 0
    for i in range(len(wavname)):
        end = end + wavlength[i]
        dict_out[wavname[i]] = output[i,:wavlength[i]]
        start = start + wavlength[i]
    return dict_out

dim = 39
maxlen = 777

if __name__ == '__main__':
    if len(sys.argv)>1:
        inputpath = sys.argv[1]
        outputname = sys.argv[2]
        wavname = outputwavname()

        path = inputpath+'mfcc/test.ark'
        testx_data = mfcc.make_input_dict(path)

        model = load_model('rnn_model_cp.h5')
        dict_out = dictwav_output(testx_data, wavname, maxlen,model)

        phone2letter = letters_39(inputpath)
        phones_39, dict48_39 = mfcc.phoneslist(inputpath+'phones/48_39.map')
        silence = phones_39.index('sil')


        f = open(outputname,'w')
        f.write('id,phone_sequence'+'\n')
        for name in wavname:
            numlist = trimming(list(dict_out[name]), silence)
            strseq = ''
            for ss in numlist:
                if ss !=39:
                    strseq = strseq + phone2letter[phones_39[ss]]
            #print(name,strseq)
            f.write(name+','+strseq+'\n')
        f.close()
