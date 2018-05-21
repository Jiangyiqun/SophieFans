import helper
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GridSearchCV

################################ debug functions ###############################
def debug(*argv):
    if True:
        print(*argv)


def debug_dict(title, my_dict):
    debug(title + ' = ')
    for key in my_dict:
        debug(key, my_dict[key])
    debug()


def debug_matrix(title, matrix):
    debug(title + ' = ')
    for line in matrix:
        debug(line)
    debug()


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
            output_vector[i] = 1
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
    weight_dict = defaultdict(float)
    for i in range(len(vocabulary)):
        weight_dict[vocabulary[i]] = weight_list[i]
    return weight_dict


def read_to_matrix(file_path):
    # read txt file to matrix
    with open(file_path,'r') as fh:
        data_matrix=[line.strip().split(' ') for line in fh]
    return data_matrix


def get_class1_vector(input_vector, weight_dict):
    # remove duplicated words
    cleaned_vector = list(set(input_vector))
    # sort based on weight (from class 1 to class 0)
    combined_vector = []
    for word in cleaned_vector:
        combined_vector.append((word, weight_dict[word]))
    sorted_vector = sorted(combined_vector,key=lambda l:l[1], reverse=True)
    # unzip and generate output_vector
    output_vector = []
    for pair in sorted_vector:
        output_vector.append(pair[0])
    return output_vector


def get_class0_vocabulary(input_vector, weight_dict, vocabulary):
    # remove words in vocabulary which are in input_vector as well
    cleaned_vector = list(set(vocabulary) - set(input_vector))
    # sort based on weight (from class 0 to class 1)
    combined_vector = []
    for word in cleaned_vector:
        combined_vector.append((word, weight_dict[word]))
    sorted_vector = sorted(combined_vector,key=lambda l:l[1])
    # unzip and generate output_vector
    output_vector = []
    for pair in sorted_vector:
        output_vector.append(pair[0])
    return output_vector


def get_modified_vector(input_vector, weight_dict, vocabulary, n):
    # change class1_vector base on class0_vocabulary
    class1_vector = get_class1_vector(input_vector, weight_dict)
    class0_vocabulary =\
            get_class0_vocabulary(input_vector, weight_dict, vocabulary)
    i_1 = 0     # counter of class1_vector
    i_0 = 0     # counter of class0_vocabulary
    output_vector = []
    # perform n time distinct change
    for i in range(n):
        # eg.   how can we get from class1 to class0
        # class1_vector[i_1]               weight_sum
        #           class0_vocabulary[i_0]              choice
        #      1           -2                 <0    add class0_vocabulary[i_0]
        #      1           -0.5               >0    rm class1_vector[i_1]
        #      1           0.5                >0    rm  class1_vector[i_1]
        #      1           2                  >0    class1_vector[i_1]
        #      -1          -2                 <0    add class0_vocabulary[i_0]
        #      -1          -0.5               <0    add class0_vocabulary[i_0]
        #      -1          0.5                <0    add class0_vocabulary[i_0]
        #      -1          2                  >0    rm  class1_vector[i_1]
        #
        weight_sum = weight_dict[class1_vector[i_1]]\
                     + weight_dict[class0_vocabulary[i_0]]
        if weight_sum < 0:
            # rm  class1_vector[i_1]
            i_1 += 1
        else:
            # add class0_vocabulary[i_0]
            output_vector.append(class0_vocabulary[i_0])
            i_0 += 1
    # add remained items to output_vector
    for i in range(i_1, len(class1_vector)):
        output_vector.append(class1_vector[i])
    return output_vector


def get_modified_matrix(input_matrix, weight_dict, vocabulary, n):
    # change from class1 to class0
    output_matrix = []
    for input_vector in input_matrix:
        output_vector = get_modified_vector(\
                input_vector, weight_dict, vocabulary, n)
        output_matrix.append(output_vector.copy())
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
    # print('class-0 prediction =\n' + str(prediction0))
    # print('class-1 prediction =\n' + str(prediction1))
    print('test_data prediction =\n' + str(prediction_test))
    # print('modiefied_data prediction =\n' + str(prediction_mod))
    # print('class-0 decision_function =\n' + str(decision_function0))
    # print('class-1 decision_function =\n' + str(decision_function1))
    print('test_data decision_function =\n' + str(decision_function_test))
    print('test_data sum = ', sum(decision_function_test))
    print('modiefied_data decision_function =\n' + str(decision_function_mod))
    print('modiefied_data sum = ', sum(decision_function_mod))

################################ fool_classifier ###############################
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()

    ########################### define parameter ###########################
    n = 20     # the number of distinct words that can be modified
    parameters={'gamma': 'auto',
                'C': 0.1,
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
    # debug_matrix('class0', strategy_instance.class0)
    # debug_matrix('class1', strategy_instance.class1)
    # get vocabulary
    vocabulary = get_vocabulary(strategy_instance)
    # debug('vocabulary =\n', vocabulary)
    # get y_train
    y_train = get_y_train(strategy_instance)
    debug('y_train =\n', y_train)
    # get x_train
    x_train = get_x_train(strategy_instance, vocabulary)
    # debug('x_train =\n', x_train)
    # training
    clf_start = strategy_instance.train_svm(parameters, x_train, y_train)
    param_range = np.arange(0.001,1,0.01)
    param_grid = [{'C': param_range, 'kernel': ['linear']}]
    grid = GridSearchCV(clf_start, param_grid)
    grid.fit(x_train,y_train)
    clf = grid.best_estimator_
    ############################# modify file ##############################
    # read test_data.txt
    test_data_matrix = read_to_matrix(test_data)
    # debug_matrix('test_data_matrix', test_data_matrix)
    # get weight_list
    weight_list = clf.coef_.tolist()[0]
    # debug('weight_list =\n', weight_list)
    # get weight_dict
    weight_dict = get_weight_dict(weight_list, vocabulary)
    # debug_dict('weight_dict', weight_dict)
    # get modified matrix
    modified_data_matrix =\
            get_modified_matrix(test_data_matrix, weight_dict, vocabulary, n)
    # debug_matrix('modified_data_matrix', modified_data_matrix)
    # write to modified_data
    modified_data='./modified_data.txt'
    write_to_file(modified_data_matrix ,modified_data)

    ################################## test  ###################################
    # Check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    # Show test result
    show_test_result(clf, vocabulary)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


################################ main function #################################
if __name__ == '__main__':
    test_data='./test_data.txt'
    strategy_instance = fool_classifier(test_data)