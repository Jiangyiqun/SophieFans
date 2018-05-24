import helper
from collections import defaultdict
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



################################ debug functions ###############################
# these functions won't be available in the submitted version
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



########################## feature extraction functions ########################
def get_x_train(strategy_instance):
    corpus = []
    for para in strategy_instance.class0 + strategy_instance.class1:
        corpus.append(' '.join(para))
    # print(corpus)
    vectorizer = TfidfVectorizer(token_pattern='\S+')
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
def read_test_matrix(file_path):
    # read txt file to matrix
    with open(file_path,'r') as fh:
        test_matrix=[line.strip().split(' ') for line in fh]
    return test_matrix


def write_modified_matrix(modified_matrix ,file_path):
    with open(file_path, 'w+') as fh:
        for modified_vector in modified_matrix:
            line = ' '.join(modified_vector) + '\n'
            fh.write(line)


def remove_all_occurrence(input_list, word):
    output_list = input_list[:]
    while True:
        try:
            output_list.remove(word)
        except ValueError:
            break
    return output_list


def get_modified_vector(test_vector, vocabulary, weight_list, vectorizer):
    to_rm_index = []   # words in test_vector
    to_rm_weight = []
    to_add_index = []  # words not in test_vector
    to_add_weight = []
    modified_vector = test_vector[:]
    # split test_vector into two kinds of lists: to_rm and to_add
    feature_vector =\
            vectorizer.transform([' '.join(test_vector)]).toarray().tolist()[0]
    for i in range(len(feature_vector)):
        if feature_vector[i] == 0.0:
            to_add_index.append(i)
            to_add_weight.append(weight_list[i])
        else:
            to_rm_index.append(i)
            to_rm_weight.append(weight_list[i])
    # debug('feature_vector=\n', feature_vector)
    # debug('to_add_index =\n', to_add_index)
    # debug('to_rm_index =\n', to_rm_index)
    # sort by weight
    to_rm_weight, to_rm_index =\
            zip(*sorted(zip(to_rm_weight,to_rm_index), reverse=True))
    to_add_weight, to_add_index =\
            zip(*sorted(zip(to_add_weight, to_add_index)))
    # remove words
    for i in range(15):
        modified_vector = remove_all_occurrence(\
                modified_vector, vocabulary[to_rm_index[i]])
    # add words
    for i in range(5):
        modified_vector.append(vocabulary[to_add_index[i]])
    return modified_vector


################################ fool_classifier ###############################
def fool_classifier(test_data): ## Please do not change the function defination...
    ## Read the test data file, i.e., 'test_data.txt' from Present Working Directory...
    ## You are supposed to use pre-defined class: 'strategy()' in the file `helper.py` for model training (if any),
    #  and modifications limit checking
    strategy_instance=helper.strategy()


    ########################### define parameter ###########################
    parameters={'gamma': 'auto',
                'C': 1.0,
                'kernel': 'linear',
                'degree': 3,
                'coef0': 0.0
                }


    ########################### feature extraction #############################
    # debug_matrix('class0', strategy_instance.class0)
    # debug_matrix('class1', strategy_instance.class1)
    y_train = get_y_train(strategy_instance)
    # debug('y_train =\n', y_train)

    x_train, vectorizer = get_x_train(strategy_instance)
    # debug('x_train =\n', x_train)


    ############################## train model #################################
    clf = strategy_instance.train_svm(parameters, x_train, y_train)
    # debug(clf)

    # grid search
    # param_range = [2**i for i in range(-5, 16)]
    # param_grid = [{'C': param_range, 'kernel': ['linear']}]
    # grid = GridSearchCV(clf_start, param_grid)
    # grid.fit(x_train,y_train)
    # clf = grid.best_estimator_

    vocabulary = vectorizer.get_feature_names()
    # debug('vocabulary =\n', vocabulary)
    
    weight_list = clf.coef_.tolist()[0]
    # debug('weight_list =\n', weight_list)


    # ############################# modify file ##############################
    modified_data='./modified_data.txt'
    #read file
    test_matrix = read_test_matrix(test_data)

    # get modified matrix
    modified_matrix = []
    for test_vector in test_matrix:
        modified_vector = get_modified_vector(\
                test_vector, vocabulary, weight_list, vectorizer)
        modified_matrix.append(modified_vector)

    # write file
    write_modified_matrix(modified_matrix, modified_data)


    ################################## test  ###################################
    # show_test_result(clf, vectorizer)
    assert strategy_instance.check_data(test_data, modified_data)

    return strategy_instance


################################ main function #################################
if __name__ == '__main__':
    test_data='./test_data.txt'
    strategy_instance = fool_classifier(test_data)