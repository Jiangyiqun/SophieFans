import helper
import numpy as np

def debug(*argv):
    if False:
        print(*argv)


def word_to_freq(word_matrix, word_table):
    # arguments: 
    #   word_matrix = [ word_list = [word]]
    #   based on word_table
    # return:
    #   freq_matrix = [ freq_list = [freq]]
    #
    total_word = len(word_table)
    freq_matrix = list()
    for word_list in word_matrix:
        # initialize freq_list with all 0
        freq_list = [0 for i in range(total_word)]
        # calculate the frequency
        for word in word_list:
            try:
                i = word_table.index(word)
                freq_list[i] += 1
            except ValueError:
                pass    # pass when the word is not in word_table
        freq_matrix.append(freq_list)
    return freq_matrix


def what_is_wrong(clf, test_path, word_table, right_answer):
    # print out the index of wrong answer
    # generate test data
    with open(test_path,'r') as test_file:
        word_matrix=[line.strip().split(' ') for line in test_file]
    freq_matrix = word_to_freq(word_matrix, word_table)
    test_matrix = np.array(freq_matrix)
    # print out the index of wrong answer
    if right_answer == 1:
        wrong_answer = 0
    else:
        wrong_answer = 1
    all_answer = clf.predict(test_matrix)
    wrong_answer = list()
    number_of_all_answer = all_answer.shape[0]
    number_of_right_answer = 0
    for i in range(number_of_all_answer):
        if all_answer[i] == right_answer:
            number_of_right_answer += 1
        else:
            wrong_answer.append(i)
    success_rate =  number_of_right_answer / number_of_all_answer * 100
    mistake = wrong_answer
    return success_rate, mistake
        
def fool_classifier(test_data, parameters): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    
    
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy() 
    # parameters={'gamma': 'auto',
    #             'C': 1.0,
    #             'kernel': 'linear',
    #             'degree': 3,
    #             'coef0': 0.0 
    #             }
    # gamma : float, optional (default='auto')
    #     Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    #     If gamma is 'auto' then 1/n_features will be used instead.
    # C : float, optional (default=1.0)
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
    ##..................................#
    # initialization
    len0 = len(strategy_instance.class0)
    len1 = len(strategy_instance.class1)

    # generate a sorted list contains all words in all samples
    word_set = set()
    for sample in strategy_instance.class0:
        for word in sample:
            word_set.add(word)
    for sample in strategy_instance.class1:
        for word in sample:
            word_set.add(word)
    word_table = sorted(list(word_set))
    total_word = len(word_table)
    # debug(word_table.index)
    debug("total_word = ", total_word)

    # generate y_train
    y_list = [0 for i in range(len0)] + [1 for i in range(len1)]
    y_train = np.array(y_list)
    debug('y_train:', type(y_train), y_train.shape)

    # generate x_train
    word_matrix = strategy_instance.class0 + strategy_instance.class1
    freq_matrix = word_to_freq(word_matrix, word_table)
    x_train = np.array(freq_matrix)
    # debug(x_train[0])
    debug('x_train:', type(x_train), x_train.shape)

    # training
    clf = strategy_instance.train_svm(parameters, x_train, y_train)
    debug(clf)

    # make prediction of test_data
    with open('./word_frequency.txt', 'a+') as line:
        line.write('\n################################################################\n')
        line.write(str(clf) + '\n')
        rate, mistake = what_is_wrong(clf, './class-0.txt', word_table, 0)
        line.write('class0 rate = ' + str(rate) + '\n')
        if mistake:
            line.write(str(mistake) + '\n')
        rate, mistake = what_is_wrong(clf, './class-1.txt', word_table, 1)
        line.write('class1 rate = ' + str(rate) + '\n')
        if mistake:
            line.write(str(mistake) + '\n')
        rate, mistake = what_is_wrong(clf, './test_data.txt', word_table, 1)
        line.write('test_data rate = ' + str(rate) + '\n')
        if mistake:
            line.write(str(mistake) + '\n')
    ##..................................#
    
    
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory...
    
    
    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    # assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


#################################### test ######################################
if __name__ == '__main__':
    test_data='./test_data.txt'
    # gamma : float, optional (default='auto')
    #     Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    #     If gamma is 'auto' then 1/n_features will be used instead.
    # C : float, optional (default=1.0)
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
    with open('./word_frequency.txt', 'a+') as line:
        line.write('############# linear kernal ##############\n')
        line.write('############# linear kernal ##############\n')
        line.write('############# linear kernal ##############\n')
    parameters={'gamma': 'auto',
            'C': 1.0,
            'kernel': 'linear',
            'degree': 3,
            'coef0': 0.0 
            }
    for i in range(-5, 16):
        c = 2 ** i
        parameters['C'] = c
        strategy_instance = fool_classifier(test_data, parameters)

    with open('./word_frequency.txt', 'a+') as line:
        line.write('############# rbf kernal ##############\n')
        line.write('############# rbf kernal ##############\n')
        line.write('############# rbf kernal ##############\n')
    parameters={'gamma': 'auto',
            'C': 1.0,
            'kernel': 'linear',
            'degree': 3,
            'coef0': 0.0 
            }
    for i in range(-5, 16):
        c = 2 ** i
        parameters['C'] = c
        for j in range(-15, 4):
            gamma = 2 ** j
            parameters['gamma'] = gamma
            strategy_instance = fool_classifier(test_data, parameters)