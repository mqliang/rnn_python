__author__ = 'starsdeep'
from rnn_theano import RNNTheano
from rnn_numpy import RNNNumpy
from gru_theano import GRUTheano

import time
import numpy as np



_VOCABULARY_SIZE = 8000
_HIDDEN_DIM = 80
_LEARNING_RATE = 0.005
_NEPOCH = 100

index_to_word = np.load('data/index_to_word.npy')
word_to_index = np.load('data/word_to_index.npy')
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')

print "load data okay, vacab size is " + str(len(index_to_word))

def test_loss(model):
    print "\ntest loss: " + str(type(model))
    print "Expected Loss for random predictions: %f" % np.log(model.word_dim)
    print "Actual loss: %f" % model.calculate_loss(X_train[:100], y_train[:100])



def test_performance(model, learning_rate):
    print "\ntest performance: " + str(type(model))
    t1 = time.time()
    model.sgd_step(X_train[10], y_train[10], learning_rate)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)



model_gru = GRUTheano(word_dim=_VOCABULARY_SIZE, hidden_dim=_HIDDEN_DIM, bptt_truncate=-1)
model_theano = RNNTheano(word_dim =_VOCABULARY_SIZE, hidden_dim=_HIDDEN_DIM, bptt_truncate=-1)
model_rnn = RNNNumpy(word_dim =_VOCABULARY_SIZE, hidden_dim=_HIDDEN_DIM, bptt_truncate=-1)

test_performance(model_gru, _LEARNING_RATE)
test_performance(model_theano, _LEARNING_RATE)
test_performance(model_rnn, _LEARNING_RATE)

test_loss(model_gru)
test_loss(model_theano)
test_loss(model_rnn)







