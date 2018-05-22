import helper
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

################################ debug functions ###############################
def debug(*argv):
    if False:
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
def get_x_train(strategy_instance):
    corpus = []
    for para in strategy_instance.class0 + strategy_instance.class1:
        corpus.append(' '.join(para))
    # print(corpus)
    vectorizer = TfidfVectorizer(binary=True, token_pattern='\S+')
    x_train = vectorizer.fit_transform(corpus).toarray()
    # print(x_train[0])
    return x_train, vectorizer



def get_y_train(strategy_instance):
    len0 = len(strategy_instance.class0)
    len1 = len(strategy_instance.class1)
    y_list = [0 for i in range(len0)] + [1 for i in range(len1)]
    y_train = np.array(y_list)
    return y_train


############################# modify file functions ############################
def get_vocabulary(vocabulary_):
    vocabulary = [None for _ in range(len(vocabulary_))]
    for token in vocabulary_:
        index = vocabulary_[token]
        vocabulary[index] =  token
    # print(vocabulary)
    return vocabulary


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


def get_feature_matrix(data_matrix, vectorizer):
    corpus = []
    for para in data_matrix:
        corpus.append(' '.join(para))
    # print(corpus)
    feature_matrix = vectorizer.transform(corpus).toarray().tolist()
    # print(x_train[0])
    return feature_matrix


def get_to_modified_words(feature_vector, weight_list, vocabulary, n):
    to_rm_candidates = []   # all index of words in feature_vector
    to_rm_weight = []
    to_add_candidates = []  # all index of words not in feature_vector
    to_add_weight = []
    to_rm_index = []        # the index need to be removed
    to_add_index = []       # the index need to be removed
    to_rm_words = []        # the word need to be removed
    to_add_words = []       # the word need to be removed
    # split data_feature into two lists
    for i in range(len(feature_vector)):
        if feature_vector[i] == 0:
            to_add_candidates.append(i)
            to_add_weight.append(weight_list[i])
        else:
            to_rm_candidates.append(i)
            to_rm_weight.append(weight_list[i])
    # debug('to_rm_candidates =\n', to_rm_candidates)
    # debug('to_rm_weight =\n', to_rm_weight)
    # debug('to_add_candidates =\n', to_add_candidates)
    # debug('to_add_weight =\n', to_add_weight)
    # sort by weight
    to_rm_weight, to_rm_candidates =\
            zip(*sorted(zip(to_rm_weight, to_rm_candidates), reverse=True))
    to_add_weight, to_add_candidates =\
            zip(*sorted(zip(to_add_weight, to_add_candidates)))
    # dynamically utilise to_rm_candidates and to_add_candidates
    i_rm = 0      # pointer in to_rm_weight & to_rm_candidates
    i_add = 0     # pointer in to_add_weight & to_add_candidates
    # perform n time distinct change
    for _ in range(n):
        #          how can we get from class1 to class0
        # to_rm_weight[i_rm]                weight_sum
        #           to_add_weight[i_add]              best choice
        #      1           -0.5               >0            rm
        #      1           0.5                >0            rm
        #      1           2                  >0            rm
        #      -1          2                  >0            rm 
        #      -1          -2                 <0            add 
        #      -1          -0.5               <0            add
        #      -1          0.5                <0            add
        #      1           -2                 <0            add
        #
        weight_sum = to_rm_weight[i_rm] + to_add_weight[i_add]
        if weight_sum > 0:
            # rm
            to_rm_index.append(to_rm_candidates[i_rm])
            i_rm += 1
        else:
            # add
            to_add_index.append(to_add_candidates[i_add])
            i_add += 1
    # change index to word
    for i in to_rm_index:
        to_rm_words.append(vocabulary[i])
    for i in to_add_index:
        to_add_words.append(vocabulary[i])
    return to_rm_words, to_add_words


def get_modified_vector(data_vector, to_rm_words, to_add_words):
    modified_vector = []
    for word in data_vector:
        if word not in to_rm_words:
            modified_vector.append(word)
    for word in to_add_words:
        modified_vector.append(word)
    return modified_vector


def get_modified_matrix(input_matrix, weight_dict, vocabulary, n):
    # change from class1 to class0
    output_matrix = []
    for input_vector in input_matrix:
        output_vector = get_modified_vector(\
                input_vector, weight_dict, vocabulary, n)
        output_matrix.append(output_vector.copy())
    return output_matrix


def write_to_file(input_matrix ,file_path):
    with open(file_path, 'w+') as fh:
        for input_vector in input_matrix:
            line = ' '.join(input_vector) + '\n'
            fh.write(line)


################################ debug functions ###############################
def get_prediction(clf, file_path, vectorizer):
    # print out the index of wrong answer
    # generate feature
    corpus = []
    with open(file_path,'r') as fh:
        for para in fh:
            corpus.append(para)
    x_test = vectorizer.transform(corpus).toarray()
    y_test = clf.predict(x_test)
    y_df = clf.decision_function(x_test)
    return y_test, y_df



def show_test_result(clf, vectorizer):
    # get prediction
    y0, df0 = get_prediction(clf, './class-0.txt', vectorizer)
    y1, df1 = get_prediction(clf, './class-1.txt', vectorizer)
    y_test,df_test = get_prediction(clf, './test_data.txt', vectorizer)
    y_mod, df_mod = get_prediction(clf, './modified_data.txt', vectorizer)
    # calculate success rate
    rate0 = y0.tolist().count(0) / y0.shape[0] * 100
    rate1 = y1.tolist().count(1) / y1.shape[0] * 100
    rate_test = y_test.tolist().count(1)/y_test.shape[0] * 100
    rate_mod = y_mod.tolist().count(0)/y_mod.shape[0] * 100
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
    # print('test_data prediction =\n' + str(prediction_test))
    # print('modiefied_data prediction =\n' + str(prediction_mod))
    # print('class-0 decision_function =\n' + str(decision_function0))
    # print('class-1 decision_function =\n' + str(decision_function1))
    print('test_data decision_function =\n' + str(df_test))
    # print('test_data sum = ', sum(decision_function_test))
    print('modiefied_data decision_function =\n' + str(df_mod))
    # print('modiefied_data sum = ', sum(decision_function_mod))

def log_change(to_rm_words, to_add_words):
    with open('./log_change.txt', 'a+') as fh:
        fh.write('Removed: ' + str(to_rm_words)\
                + "   Added: " + str(to_add_words) + '\n')


################################ fool_classifier ###############################
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()

    ########################### define parameter ###########################
    n = 20     # the number of distinct words that can be modified
    parameters={'gamma': 'auto',
                'C': 4.0,
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

    ############################# train dkata ###############################
    # debug_matrix('class0', strategy_instance.class0)
    # debug_matrix('class1', strategy_instance.class1)
    y_train = get_y_train(strategy_instance)
    debug('y_train =\n', y_train)

    x_train, vectorizer = get_x_train(strategy_instance)
    debug('x_train =\n', x_train)

    # training
    clf = strategy_instance.train_svm(parameters, x_train, y_train)
    # debug(clf)

    # grid search
    # param_range = [2**i for i in range(-5, 16)]
    # param_grid = [{'C': param_range, 'kernel': ['linear']}]
    # grid = GridSearchCV(clf_start, param_grid)
    # grid.fit(x_train,y_train)
    # clf = grid.best_estimator_

    # ############################# modify file ##############################
    vocabulary = get_vocabulary(vectorizer.vocabulary_)
    debug('vocabulary =\n', vocabulary)

    weight_list = clf.coef_.tolist()[0]
    debug('weight_list =\n', weight_list)

    deanna = sorted(zip(weight_list, vocabulary))
    with open('./deanna.txt', 'w+') as fh:
        for line in deanna:
            fh.write(str(line[1]) + ': ' + str(line[0]) + '\n')


    data_matrix = read_to_matrix(test_data)
    # debug_matrix('data_matrix =\n', data_matrix)

    feature_matrix = get_feature_matrix(data_matrix, vectorizer)
    # debug_matrix('feature_matrix =\n', feature_matrix)

    # get modified matrix
    modified_matrix = []
    for i in range(len(data_matrix)):
        data_vector = data_matrix[i]
        feature_vector = feature_matrix[i]
        to_rm_words, to_add_words =\
                get_to_modified_words(feature_vector, weight_list, vocabulary, n)
        # debug('to_rm_words', to_rm_words)
        # debug('to_add_words', to_add_words)
        modified_vector = get_modified_vector(data_vector,\
                to_rm_words, to_add_words)
        # debug('modified_vector =\n', modified_vector)
        modified_matrix.append(modified_vector.copy())
    # debug_matrix('modified_matrix', modified_matrix)

    # write to modified_data
    modified_data='./modified_data.txt'
    write_to_file(modified_matrix ,modified_data)

    ################################## test  ###################################
    # Check that the modified text is within the modification limits.
    assert strategy_instance.check_data(test_data, modified_data)
    # Show test result
    show_test_result(clf, vectorizer)
    return strategy_instance ## NOTE: You are required to return the instance of this class.


################################ main function #################################
if __name__ == '__main__':
    test_data='./test_data.txt'
    strategy_instance = fool_classifier(test_data)