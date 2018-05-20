import helper
import numpy as np

def debug(*argv):
    if False:
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


def get_feature(input_matrix, vocabulary):
    # arguments: 
    #   input_matrix = [ input_list = [word]]
    # return:
    #   output_matrix = [ output_list = [existence of word]]
    #   based on vocabulary
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
                output_list[i] = 1
            except ValueError:
                pass    # pass when the word is not in word_table
        output_matrix.append(output_list)
    return output_matrix


def get_x_train(strategy_instance, vocabulary):
    word_matrix = strategy_instance.class0 + strategy_instance.class1
    x_matrix = get_feature(word_matrix, vocabulary)
    x_train = np.array(x_matrix)
    return x_train


def get_y_train(strategy_instance):
    len0 = len(strategy_instance.class0)
    len1 = len(strategy_instance.class1)
    y_list = [0 for i in range(len0)] + [1 for i in range(len1)]
    y_train = np.array(y_list)
    return y_train


def get_prediction(clf, file_path, vocabulary):
    # print out the index of wrong answer
    # generate feature
    with open(file_path,'r') as fh:
        word_matrix=[line.strip().split(' ') for line in fh]
    x_matrix = get_feature(word_matrix, vocabulary)
    x_train = np.array(x_matrix)
    prediction = clf.predict(x_train)
    return prediction


def print_test(clf, vocabulary):
    dim = len(vocabulary)
    # get prediction
    prediction0 = get_prediction(clf, './class-0.txt', vocabulary)
    prediction1 = get_prediction(clf, './class-1.txt', vocabulary)
    prediction_test = get_prediction(clf, './test_data.txt', vocabulary)
    prediction_mod = get_prediction(clf, './modified_data.txt', vocabulary)
    # calculate success rate
    rate0 = prediction0.tolist().count(0) / prediction0.shape[0] * 100
    rate1 = prediction1.tolist().count(1) / prediction1.shape[0] * 100
    rate_test = prediction_test.tolist().count(1)/prediction_test.shape[0] * 100
    rate_mod = prediction_mod.tolist().count(0)/prediction_mod.shape[0] * 100
    # print test result
    print('########################## Test Result ############################')
    print(str(clf))
    print('############################ Summery ##############################')
    print('class-0 Success Rate = ' + str(rate0))
    print('class-1 Success Rate = ' + str(rate1))
    print('test_data Success Rate = ' + str(rate_test))
    print('modified_data Success Rate = ' + str(rate_mod))
    print('############################# Detail ##############################')
    print('class-0 prediction =\n' + str(prediction0))
    print('class-1 prediction =\n' + str(prediction1))
    print('test_data prediction =\n' + str(prediction_test))
    print('modiefied_data prediction =\n' + str(prediction_test))



def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()

    ############################# define parameter #############################
    parameters={'gamma': 'auto',
                'C': 1.0,
                'kernel': 'linear',
                'degree': 3,
                'coef0': 0.0
                }
    # gamma : float, optional (default='auto') 2^-15 ~ 2^3
    #     Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    #     If gamma is 'auto' then 1/n_features will be used instead.
    # C : float, optional (default=1.0) 2^-5 ~ 2^15
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

    ############################### train data #################################
    debug_matrix('class0', strategy_instance.class0)
    debug_matrix('class1', strategy_instance.class1)
    # get vocabulary
    vocabulary = get_vocabulary(strategy_instance)
    debug('vocabulary =\n', vocabulary)
    # get y_train
    y_train = get_y_train(strategy_instance)
    debug('y_train =\n', y_train)
    # get x_train
    x_train = get_x_train(strategy_instance, vocabulary)
    debug('x_train =\n', x_train)
    # training
    clf = strategy_instance.train_svm(parameters, x_train, y_train)

    ############################### modify file ################################
    # generate weight_table corresponded by vocabulary
    # weight_table = clf.coef_.tolist()[0]
    # class0_vocabulary = []
    # for i in range(2):
    #     class0_word = vocabulary[weight_table.index(sorted(weight_table)[i])]
    #     class0_vocabulary.append(class0_word)
    # class1_vocabulary = []
    # for i in range(2):
    #     class1_word = vocabulary[weight_table.index(sorted(weight_table)[-i - 1])]
    #     class1_vocabulary.append(class1_word)
    # # debug(vocabulary)
    # # debug(weight_table)
    # debug(class0_vocabulary)
    # debug(class1_vocabulary)
    # # read file
    # with open('./test_data.reduced','r') as test_data_file:
    #     test_data_matrix=[line.strip().split(' ') for line in test_data_file]
    # # debug(test_data_matrix)
    # # generate weight by index
    # # exchange_data_matrix = [
    # #       sample_exchange_data = [ 
    # #           word_exchange_data = [
    # #               [weight, index, word_test_data]]]]
    # # Note: word_exchange_data is sorted by weight
    # exchange_data_matrix = []
    # for sample_test_data in test_data_matrix:
    #     sample_exchange_data = []
    #     for index in range(len(sample_test_data)):
    #         word_test_data = sample_test_data[index]
    #         try:
    #             weight = weight_table[vocabulary.index(word_test_data)]
    #         except ValueError:
    #             weight = 0
    #         word_exchange_data = [weight, index, word_test_data]
    #         sample_exchange_data.append(word_exchange_data)
    #     sample_exchange_data = sorted(sample_exchange_data,key=lambda l:l[0])
    #     exchange_data_matrix.append(sample_exchange_data)
    # # debug(exchange_data_matrix)
    # # generate the index of class1 feature word
    # class1_feature_word_matrix = []
    # for vector_exchange in exchange_data_matrix:
    #     vector_feature = []
    #     for word_exchange in vector_exchange[0:2]:
    #         # vector_feature.append(word_exchange[1])
    #         pass
    #     class1_feature_word_matrix.append(vector_feature)
    # # debug(class1_feature_word_matrix)
    # # write modified data
    # with open('./modified_data.txt', 'w') as modiefied_data_file:
    #     for i in range(len(test_data_matrix)):
    #         modiefied_data_list = test_data_matrix[i][:]
    #         for j in range(len(class1_feature_word_matrix[i])):
    #             index = class1_feature_word_matrix[i][j]
    #             modiefied_data_list[j] = class0_vocabulary[i]
    #         modiefied_data_str = ' '.join(modiefied_data_list)
    #         # debug(modiefied_data_str)
    #         modiefied_data_file.write(modiefied_data_str + '\n')

    ## You can check that the modified text is within the modification limits.
    modified_data='./modified_data.txt'
    # assert strategy_instance.check_data(test_data, modified_data)
    print_test(clf, vocabulary)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


#################################### test ######################################
if __name__ == '__main__':
    test_data='./test_data.txt'
    strategy_instance = fool_classifier(test_data)