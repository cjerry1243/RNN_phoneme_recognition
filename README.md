# RNN Phoneme Recognition
Use recurrent neural networks for speech recognition in TIMIT.

CRNN is also used in this program.

Model with masking layer gets the best results, average 10 insertion + deletion + substitution error in a sentence.

## Environment:
python 3

Keras==2.0.7

Tensorflow==1.3


## Test:
sh (rnn.sh/cnn.sh/best.sh) current_directory output_filename

phoneme sequence will be produced in output_filename

