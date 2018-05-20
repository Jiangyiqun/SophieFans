import helper
import time, datetime
import numpy as np
from math import log
from sklearn import preprocessing


def debug(*argv):
    if True:
        print(*argv)


def debug_matrix(title, matrix):
    if False:
        print(title + ' = ')
        for line in matrix:
            print(line)


def get_vocabulary(strategy_instance):
    # generate a sorted list contains all words in all samples
    word_set = set()
    for sample in strategy_instance.class0:
        for word in sample:
            word_set.add(word)
    for sample in strategy_instance.class1:
        for word in sample:
            word_set.add(word)
    vocabulary = sorted(list(word_set))
    return vocabulary


def get_raw_freq(input_matrix, vocabulary):
    # arguments: 
    #   input_matrix = [ input_list = [word]]
    #   based on word_table
    # return:
    #   output_matrix = [ output_list = [freq]]
    #
    dim = len(vocabulary)
    output_matrix = list()
    for input_list in input_matrix:
        # initialize freq_list with all 0
        output_list = [0 for i in range(dim)]
        # calculate the frequency
        for word in input_list:
            try:
                i = vocabulary.index(word)
                output_list[i] += 1
            except ValueError:
                pass    # pass when the word is not in word_table
        output_matrix.append(output_list)
    return output_matrix


def get_log_freq(input_matrix, vocabulary):
    dim = len(vocabulary)
    output_matrix = list()
    for input_list in input_matrix:
        output_list = [0 for _ in range(dim)]
        for i in range(dim):
            output_list[i] = log((1.0 + input_list[i]), 10)
        output_matrix.append(output_list)
    return output_matrix


def get_feature(input_matrix, vocabulary, do_log, do_norm):
    # do_log: True of False
    # do_norm: None, 'l1' or 'l2'
    #
    # debug('do_log = ', do_log, '   do_norm = ', do_norm)
    # get raw frequency matrix
    debug_matrix('word_matrix', input_matrix)
    output_matrix = get_raw_freq(input_matrix, vocabulary)
    debug_matrix('freq_matrix', output_matrix)
    # perform logarithmic
    if do_log:  
        output_matrix = get_log_freq(output_matrix, vocabulary)
        debug_matrix('log_matrix', output_matrix)
    # perform normalization 
    if do_norm: 
        output_matrix = preprocessing.normalize(output_matrix, norm=do_norm)
        debug_matrix('norm_matrix', output_matrix)
    return output_matrix


def get_x_train(strategy_instance, vocabulary, do_log, do_norm):
    word_matrix = strategy_instance.class0 + strategy_instance.class1
    feature_matrix = get_feature(word_matrix, vocabulary, do_log, do_norm)
    x_train = np.array(feature_matrix)
    return x_train


def get_y_train(strategy_instance):
    len0 = len(strategy_instance.class0)
    len1 = len(strategy_instance.class1)
    y_list = [0 for i in range(len0)] + [1 for i in range(len1)]
    y_train = np.array(y_list)
    return y_train


def get_prediction(clf, feature_file_path, vocabulary, do_log, do_norm):
    # print out the index of wrong answer
    # generate feature
    with open(feature_file_path,'r') as feature_file:
        word_matrix=[line.strip().split(' ') for line in feature_file]
    feature_matrix = get_feature(word_matrix, vocabulary, do_log, do_norm)
    x_feature = np.array(feature_matrix)
    prediction = clf.predict(x_feature)
    return prediction


def log_result(clf, vocabulary, do_log, do_norm):
    dim = len(vocabulary)
    prediction0 = get_prediction(clf, './class-0.txt', vocabulary,\
            do_log, do_norm)
    rate0 = prediction0.tolist().count(0) / prediction0.shape[0] * 100
    prediction1 = get_prediction(clf, './class-1.txt', vocabulary,\
            do_log, do_norm)
    rate1 = prediction1.tolist().count(1) / prediction1.shape[0] * 100
    prediction_test = get_prediction(clf, './test_data.txt', vocabulary,\
            do_log, do_norm)
    rate_test = prediction_test.tolist().count(1) / prediction_test.shape[0] * 100
    with open('./rbf_l2norm.txt', 'a+') as handle:
        handle.write('######################################################\n')
        handle.write('PPR(do_log=' + str(do_log) + ', do_norm ='\
                + str(do_norm) + ')\n')
        handle.write(str(clf) + '\n')
        handle.write('class-0   Success Rate = ' + str(rate0) + '\n')
        handle.write('class-1   Success Rate = ' + str(rate1) + '\n')
        handle.write('test_data Success Rate = ' + str(rate_test) + '\n')
        if rate0 < 100:
            handle.write('class-0 =\n' + str(prediction0) + '\n')
        if rate1 < 100:
            handle.write('class-1 =\n' + str(prediction1) + '\n')
        if rate_test < 100:
            handle.write('test_data =\n' + str(prediction_test) + '\n')
        handle.write('\n')


def calculate_parameter(strategy_instance, vocabulary, parameters,\
        y_train, do_log, do_norm, remained_iteration):
    time_start = time.time()
    x_train = get_x_train(strategy_instance, vocabulary, do_log, do_norm)
    clf = strategy_instance.train_svm(parameters, x_train, y_train)
    log_result(clf, vocabulary, do_log, do_norm)
    time_end = time.time()
    print("Still need ", datetime.timedelta(seconds = time_end - time_start)\
            * (remained_iteration - 1))

def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy() 

    # gamma : float, optional (default='auto') 2**-15 ~ 2**3
    #     Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    #     If gamma is 'auto' then 1/n_features will be used instead.
    # C : float, optional (default=1.0) 2**-5 ~ 2**15
    #     Penalty parameter C of the error term.
    # kernel : string, optional (default='rbf')
    #     Specifies the kernel type to be used in the algorithm.
    #     It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
    #     a callable.
    #     If none is given, 'rbf' will be used. If a callable is given it is
    #     used to pre-compute the kernel matrix from data matrices; that matrix
    #     should be an array of shape ``(n_samples, n_samples)``.
    # degree : int, optional (default=3)
    #     Degree of the polynomial kernel function ('poly').
    #     Ignored by all other kernels.
    # coef0 : float, optional (default=0.0)
    #    Independent term in kernel function.
    #    It is only significant in 'poly' and 'sigmoid'.
    parameters={'gamma': 'auto',
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'coef0': 0.0 
            }
    do_log = False
    do_norm = 'l2'
    ##..................................#
    # get vocabulary
    vocabulary = get_vocabulary(strategy_instance)
    # get y_train
    y_train = get_y_train(strategy_instance)
    # compare different parameters
    remained_iteration = 798
    do_log = False
    for i in range(-5, 16):
        c = 2 ** i
        parameters['C'] = c
        for j in range(-15, 4):
            parameters['gamma'] = 2 ** j
            calculate_parameter(strategy_instance, vocabulary, parameters,\
                y_train, do_log, do_norm, remained_iteration)
            remained_iteration -= 1
    do_log = True
    for i in range(-5, 16):
        c = 2 ** i
        parameters['C'] = c
        for j in range(-15, 4):
            parameters['gamma'] = 2 ** j
            calculate_parameter(strategy_instance, vocabulary, parameters,\
                y_train, do_log, do_norm, remained_iteration)
            remained_iteration -= 1
    # debug('y_train:', type(y_train), y_train.shape)
    # get x_train
    # x_train = get_x_train(strategy_instance, do_log=False, do_norm=None)
    # x_train = get_x_train(strategy_instance, do_log=False, do_norm='l1')
    # x_train = get_x_train(strategy_instance, do_log=False, do_norm='l2')
    # x_train = get_x_train(strategy_instance, do_log=True, do_norm=None)
    # x_train = get_x_train(strategy_instance, do_log=True, do_norm='l1')
    # x_train = get_x_train(strategy_instance, vocabulary, do_log, do_norm)
    # debug('x_train:', type(x_train), x_train.shape)
    # clf = strategy_instance.train_svm(parameters, x_train, y_train)
    # log_result(clf, vocabulary, do_log, do_norm)
    ##..................................#
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    # assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


#################################### test ######################################
if __name__ == '__main__':
    test_data='./test_data.txt'
    strategy_instance = fool_classifier(test_data)