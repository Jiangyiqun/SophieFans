import helper
import numpy as np


################################ debug functions ###############################
def debug(*argv):
    if False:
        print(*argv)
        print()

def debug_matrix(title, matrix):
    if False:
        print(title + ' = ')
        for line in matrix:
            print(line)
        print()

############################## train data functions ############################
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


def get_feature_vector(input_vector, vocabulary):
    dim = len(vocabulary)
    output_vector = [0 for i in range(dim)]
    for word in input_vector:
        try:
            i = vocabulary.index(word)
            output_vector[i] += 1
        except ValueError:
            pass    # remains 0 when the word is not in word_table
    return output_vector


def get_feature_matrix(input_matrix, vocabulary):
    # arguments: 
    #   input_matrix = [ input_list = [word]]
    # return:
    #   output_matrix = [ output_list = [existence of word]]
    #   based on vocabulary
    #
    output_matrix = list()
    for input_vector in input_matrix:
        output_vector = get_feature_vector(input_vector, vocabulary)
        output_matrix.append(output_vector)
    return output_matrix


def get_x_train(strategy_instance, vocabulary):
    word_matrix = strategy_instance.class0 + strategy_instance.class1
    x_matrix = get_feature_matrix(word_matrix, vocabulary)
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
    x_matrix = get_feature_matrix(word_matrix, vocabulary)
    x_train = np.array(x_matrix)
    prediction = clf.predict(x_train)
    decision_function = clf.decision_function(x_train)
    return prediction, decision_function


############################# modify file functions ############################
def get_weight_dict(weight_list, vocabulary):
    weight_dict = dict()
    for i in range(len(vocabulary)):
        weight_dict[vocabulary[i]] = weight_list[i]
    return weight_dict


def get_class_vocabulary(weight_list, vocabulary, number):
    # get the most distinctive words of vocabulary
    # weight for class 0 is more negative
    # weight for class 1 is more positvie
    #
    # combine two lists
    dim = len(vocabulary)
    combined_list = []
    for i in range(dim):
        combined_list.append([vocabulary[i], weight_list[i]])
    # sort by weight
    sorted_list = sorted(combined_list,key=lambda l:l[1])
    # generate class_words
    class0_vocabulary = []
    class1_vocabulary = []
    for i in range(dim):
        class0_vocabulary.append(sorted_list[i][0])
        class1_vocabulary.append(sorted_list[-1 - i][0])
    return class0_vocabulary, class1_vocabulary



def read_to_matrix(file_path):
    # read txt file to matrix
    with open(file_path,'r') as fh:
        data_matrix=[line.strip().split(' ') for line in fh]
    return data_matrix


def get_class_word_matrix(weight_dict, input_matrix, number):
    # get the most distinctive words of input_matrix without duplicated words
    # weight for class 0 is more negative
    # weight for class 1 is more positvie
    #
    # combine sample with weight
    class0_word_matrix = []
    class1_word_matrix = []
    for input_vector in input_matrix:
        # combine input_vector with weight
        combined_vector = []
        for word in set(input_vector):  # remove duplicated words
            try:
                combined_vector.append([word, weight_dict[word]])
            except KeyError:    # give 0.0 to words that are not in vocabulary
                combined_vector.append([word, 0.0])
        # sort by weight
        sorted_vector = sorted(combined_vector,key=lambda l:l[1])
        # generate class0_vector
        class0_vector = []
        for word in sorted_vector:
            class0_vector.append(word[0])
        class0_word_matrix.append(class0_vector)
        # generate class1_vector
        class1_vector = []
        for word in reversed(sorted_vector):
            class1_vector.append(word[0])
        class1_word_matrix.append(class1_vector)
    return class0_word_matrix, class1_word_matrix


def replace_all_occurrence(input_list, find, replace):
    # replace find (a word, all occurrence) in input_list to replace (a word)
    output_list = input_list[:]
    for i in range(len(input_list)):
        if input_list[i] == find:
            output_list[i] = replace
            for _ in range(10):
                output_list.append(replace)
        else:
            pass    # remains equal to input_list[i]
    return output_list


def get_modified_matrix(input_matrix, class_word_matrix,\
        class_vocabulary, number):
    # Input and Return format:
    #   input_matrix: [input_vector: [input_word]]
    #   output_matrix: [output_vector: [output_word]]
    #
    # Description:
    #   modify number of input_word which are in find_vector 
    # to different words which are in replace_vector.
    #   so that output_matrix can be more like class in class_vocabulary
    # rather than class in.
    
    output_matrix = []
    # generate output_matrix
    for i_line in range(len(input_matrix)):
        # generate input_vector
        input_vector = input_matrix[i_line]
        # generate find_vector:
        #   - find_vector is derived from class_word_matrix
        find_vector = class_word_matrix[i_line]
        # generate replace_vector:
        #   - derived from class_vocabulary
        #   - has number of words
        #   - all words must not in input_vector
        replace_vector = []
        i_replace = 0
        i_vocabulary = 0
        while (i_replace < number):
            if class_vocabulary[i_vocabulary] in input_vector:
                i_vocabulary += 1
            else:
                replace_vector.append(class_vocabulary[i_vocabulary])
                i_vocabulary += 1
                i_replace +=1
        # generate output_vector
        output_vector = input_vector[:]
        for i_word in range(number):
            # replace words in find_vector to words in replace_vector
            output_vector= replace_all_occurrence(output_vector,\
                    find_vector[i_word], replace_vector[i_word])
        output_matrix.append(output_vector)
    # print('input_matrix = ', input_matrix, '\n')
    # print('class_word_matrix = ', class_word_matrix, '\n')
    # print('output_matrix = ', output_matrix, '\n')
    return output_matrix


def write_to_file(input_matrix ,file_path):
    with open(file_path, 'w') as fh:
        for input_vector in input_matrix:
            line = ' '.join(input_vector) + '\n'
            fh.write(line)


################################ debug functions ###############################
def show_test_result(clf, vocabulary):
    dim = len(vocabulary)
    # get prediction
    prediction0, decision_function0 = get_prediction(clf, './class-0.txt', vocabulary)
    prediction1, decision_function1 = get_prediction(clf, './class-1.txt', vocabulary)
    prediction_test,decision_function_test = get_prediction(clf, './test_data.txt', vocabulary)
    prediction_mod, decision_function_mod = get_prediction(clf, './modified_data.txt', vocabulary)
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
    print('modiefied_data prediction =\n' + str(prediction_mod))
    print('class-0 decision_function =\n' + str(decision_function0))
    print('class-1 decision_function =\n' + str(decision_function1))
    print('test_data decision_function =\n' + str(decision_function_test))
    print('modiefied_data decision_function =\n' + str(decision_function_mod))


################################ fool_classifier ###############################
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()

    ########################### define parameter ###########################
    n = int(10)     
    # the number of words need to be updated
    # which is 1/2 number of modification
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

    ############################# train data ###############################
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

    ############################# modify file ##############################
    # change test_data from class1 to class0
    # get weight_list
    weight_list = clf.coef_.tolist()[0]
    # debug('weight_list =\n', weight_list)
    # get weight_dict
    weight_dict = get_weight_dict(weight_list, vocabulary)
    debug('weight_dict =\n', weight_dict)
    # get_class_vocabulary
    class0_vocabulary, class1_vocabulary =\
            get_class_vocabulary(weight_list, vocabulary, n)
    debug('class0_vocabulary =\n', class0_vocabulary)
    debug('class1_vocabulary =\n', class1_vocabulary)
    # read test_data.txt
    test_data_matrix = read_to_matrix(test_data)
    debug_matrix('test_data_matrix', test_data_matrix)
    # get_class_word_matrix
    class0_word_matrix, class1_word_matrix =\
            get_class_word_matrix(weight_dict, test_data_matrix, n)
    debug_matrix('class0_word_matrix', class0_word_matrix)
    debug_matrix('class1_word_matrix', class1_word_matrix)
    # get modified matrix
    modified_data_matrix = get_modified_matrix(test_data_matrix,\
            class1_word_matrix, class0_vocabulary, n)
    debug_matrix('modified_data_matrix', modified_data_matrix)
    # write to modified_data
    modified_data='./modified_data.txt'
    write_to_file(modified_data_matrix ,modified_data)

    ################################## test  ###################################
    # Check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    # Show test result
    # show_test_result(clf, vocabulary)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


################################ main function #################################
if __name__ == '__main__':
    test_data='./test_data.txt'
    strategy_instance = fool_classifier(test_data)