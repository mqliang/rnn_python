__author__ = 'starsdeep'


if __name__ == '__main__':

    #this data is produced by script data_preprocess
    # X_train array of list, a list represent a sentence, y_train is the origin sentence shifted to right by 1 word.
    # sentence[0]: [12,31,3234,42,53]
    #X_train[0]: [12,31,3234,42]
    #y_train[1]: [31,3234,42,53]

    index_to_word = np.load('data/index_to_word.npy')
    word_to_index = np.load('data/word_to_index.npy')
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    vocabulary_size = len(index_to_word)

    print "vocabulary size: %d" % vocabulary_size
    # test cross entropy loss
    print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
