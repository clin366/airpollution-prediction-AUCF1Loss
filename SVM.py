import numpy as np
import pandas as pd
from keras.layers import Input
from keras.regularizers import l2
from keras.layers import Dense,concatenate,Dropout,Conv1D,Flatten,MaxPooling1D
from keras.models import Model,model_from_json
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score 
from dataPreprocess import DataSplit
import pickle
import argparse
import xlwt
import xlrd
import xlutils.copy
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


"""
Command line: 
python3 CNN_model.py -f ./res/keywords_data_rescaled_joined.csv -fo CNN_res_glove_summary.xls
python3 CNN_model.py -f ./res/keywords_data_rescaled_joined.csv -fo CNN_search_lag.xls
python3 CNN_model.py -f ./res/keywords_data_rescaled_joined.csv -fo CNN_search_lag_pol60.xls

"""

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    # columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    df = np.array(df)
    return df

def generate_input_sequence(supervised_values, seq_length = 5):
    embedding_dim = supervised_values.shape[1]
    na_vec = np.array([0. for i in range(embedding_dim)])
    input_embedding = []
    for i in range(len(supervised_values)):
        input_series = []
        for days_index in range(i-seq_length+1, i+1):
            if days_index >= 0:
                day_embedding = supervised_values[days_index]
            else:
                day_embedding = na_vec.copy()

            input_series.append(day_embedding)
        input_embedding.append(np.array(input_series))
    input_embedding = np.array(input_embedding)
    return input_embedding

def keras_cnn_model(seq_length = 5, embedding_dim = 5, first_ksize = 2):
    # parameters of CNN network
    reg = l2(0.08)
    # first_ksize = 2

    # model structure 
    input1 = Input(shape=(seq_length, embedding_dim))

    first_kernel = Conv1D(8, kernel_size = first_ksize, strides = 1, padding='valid', activation = 'relu')(input1)
    first_kernel = MaxPooling1D(pool_size=(seq_length-first_ksize+1), strides=(1), padding='valid')(first_kernel)
    first_dense = Flatten()(first_kernel)
    output = Dense(1,activation='sigmoid', activity_regularizer=reg)(first_dense)

    model = Model(inputs=input1, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def generate_feature_embeddings(X_train, word_embeddings, weighted = False):
    all_columns = list(X_train.columns)
    embed_keys = list(word_embeddings.keys())
    terms_column = [i for i in all_columns if i.lower() in embed_keys]
    # terms_column = list(X_train.columns)
    feature_embeddings = []
    one_hot_dim = len(word_embeddings['ozone'])

    for row_index in range(len(X_train)):
        feature_embedding = np.array([0. for i in range(0, one_hot_dim)])
        day_search = X_train.iloc[row_index,:]
        weights = []
        for i in terms_column:
            weights.append(day_search[i])

        if weighted:
            weights_sum = sum(weights)
        else:
            weights_sum = 1.0

        for i in range(len(terms_column)):
            word = terms_column[i].lower()
            word_weight = weights[i]
            word_embedding = np.array(word_embeddings[word])
            feature_embedding = feature_embedding + word_embedding * (word_weight/weights_sum)
            # feature_embedding = feature_embedding + word_embedding * word_weight

        feature_embeddings.append(feature_embedding)
    return np.array(feature_embeddings)

def generate_one_hot_embedding(embedding_dict_path, X_concat_frames, weighted = False):
    with open(embedding_dict_path, 'rb') as handle:
        word_embeddings = pickle.load(handle)
    feature_embeddings = generate_feature_embeddings(X_concat_frames, word_embeddings, weighted = weighted)
    return feature_embeddings

def run_keras_model(model, input_embedding, y_class, train_len, valid_len, test_len):
    # split data into train and test-sets
    x_train, x_valid, x_test = input_embedding[0:train_len], input_embedding[train_len:train_len+valid_len], \
                            input_embedding[train_len+valid_len:]
    y_train, y_valid, y_test  = y_class[0:train_len], y_class[train_len:train_len+valid_len], \
                            y_class[train_len+valid_len:]

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    # calculate the weights
    (_, train_count) = np.unique(y_train, return_counts=True)
    (_, valid_count) = np.unique(y_valid, return_counts=True)
    # sum_weights = float(train_count[1] + valid_count[1]) + float(train_count[0] + valid_count[0])

    class_weight = {0: float(train_count[1] + valid_count[1]), \
                    1: float(train_count[0] + valid_count[0])}

    history = model.fit(x_train, y_train, batch_size = 16, epochs = 1000, validation_data = (x_valid, y_valid), class_weight = class_weight, verbose=1, callbacks=[es])
    best_epoch = len(history.epoch)
    model.fit(x_valid, y_valid, batch_size = 16, epochs = best_epoch,class_weight = class_weight, verbose=1)
    pred = model.predict(x_test)
    pred_class = [0 if i < 0.5 else 1 for i in pred]

    # evaluation results
    accuracy = accuracy_score(y_test, pred_class)
    f1_value = f1_score(y_test, pred_class)
    fpr,tpr,threshold = roc_curve(y_test, pred) 
    auc_value = auc(fpr,tpr) 
    return accuracy, f1_value, auc_value
    
def generate_search_embedding(X_concat_frames, representation = 'one-hot'):
    if representation == 'one-hot' :
        embedding_dict_path = './res/one_hot_embeddings.pkl'
        feature_embeddings = generate_one_hot_embedding(embedding_dict_path, X_concat_frames, weighted = False)
        feature_embeddings -= np.mean(feature_embeddings, axis = 0) # zero-center
        feature_embeddings /= np.std(feature_embeddings, axis = 0) # normalize
        return feature_embeddings
    elif representation == 'glove':
        glove_dict_path = './res/glove_embeddings.pkl'
        glove_feature_embeddings = generate_one_hot_embedding(glove_dict_path, X_concat_frames, weighted = False)
        # glove_feature_embeddings = generate_one_hot_embedding(glove_dict_path, X_concat_frames, weighted = True)
        glove_feature_embeddings -= np.mean(glove_feature_embeddings, axis = (0,1)) # zero-center
        glove_feature_embeddings /= np.std(glove_feature_embeddings, axis = (0,1)) # normalize
        return glove_feature_embeddings

def lag_search_features(feature_embeddings, lag = 0):
    embedding_dim = feature_embeddings.shape[1]
    na_embedding = np.array([0. for i in range(embedding_dim)])
    reveserse_embeddings = feature_embeddings[::-1]
    lag_features = np.roll(reveserse_embeddings, lag, axis=0)
    for i in range(lag):
        lag_features[i] = na_embedding
    lag_features = lag_features[::-1]
    return lag_features

def main(file_in, file_out):
    # file_in = '../Re__Research_on_detecting_air_pollution_related_terms_searches_/keywords_data_rescaled_joined.csv'
    # air_data_raw = readData(file_in)

    # create an excel book
    book = xlwt.Workbook() 
    sheet0 = book.add_sheet('first_page')
    book.save(file_out)

    parameters = []
    # for lag_days in [3, 5, 7]: 
    #     for kernel_size in range(2, lag_days):
    #         for pollution_value in [60]:
    #             for search_lag in [0, 1, 2, 3]:
    #                 parameters.append((lag_days, kernel_size, pollution_value, search_lag))

    '''============Summary: 2009 90==============
    no polluted days in training data
    '''
    for lag_days in [7]: 
        for kernel_size in [2]:
            for pollution_value in [70]:
                for search_lag in [2]:
                    parameters.append((lag_days, kernel_size, pollution_value, search_lag))

    for parameter_index in range(len(parameters)):
        data = xlrd.open_workbook(file_out)
        ws = xlutils.copy.copy(data)
        data.release_resources()
        del data
        lag_days, kernel_size, pollution_value, search_lag= parameters[parameter_index]
        seq_length = lag_days

        sheet1 = ws.add_sheet('model' + str(parameter_index))
        row_index = 0
        col_index = 0
        
        sheet1.write(row_index,col_index,'Input_Features') 
        col_index = col_index + 1
        sheet1.write(row_index,col_index,'Accuracy') 
        col_index = col_index + 1
        sheet1.write(row_index,col_index, 'F1_score')
        col_index = col_index + 1
        sheet1.write(row_index,col_index, 'AUC_val')
        col_index = col_index + 1
        sheet1.write(row_index,col_index+2, 'CNN: ' + '(seq_length, kernel_size, pollution_value, search_lag):' + str(parameters[parameter_index]))
        col_index=0
        row_index = row_index + 1

        # with open(file_out, 'w') as fo:
        # fo.write('Input_Features'+',' + 'Accuracy'+ ',' + 'F1_score' + ',' + 'AUC_val' + '\n')
        for season in ['summer']:
        # for season in ['summer', 'winter']:
            sheet1.write(row_index, col_index, "============" + season + "=============")
            row_index = row_index + 1
            # fo.write("============" + season + "============="+ '\n')
            for final_year in [2009,2010,2011,2012]:
            # for final_year in [2009]:
                sheet1.write(row_index, col_index, 'Final year: ' + str(final_year))
                row_index = row_index + 1
                # fo.write('Final year: ' + str(final_year) + '\n')
                # air_data = selectData(air_data_raw.copy(), season = season, final_year=final_year)
                for shift_days in [0]:
                    # fo.write('Shift days: ' + str(shift_days)+ '\n')
                    print("============Summary: " + str(final_year) + ' ' + str(pollution_value) + '==============' )
                    single_feature = False
                    data_split = DataSplit(file_path = file_in, season = season, final_year = final_year)
                    X_train, X_valid, X_test, y_train, y_valid, y_test = data_split.generateTrainTest()
                    train_len = len(y_train)
                    valid_len = len(y_valid)
                    test_len = len(y_test)

                    # lag_days = 3
                    # seq_length = 3
                    # kernel_size = 2
                    # pollution_value = 50

                    raw_values = np.concatenate((y_train, y_valid, y_test), axis=0)
                    # transform data to be supervised learning
                    # supervised_values = timeseries_to_supervised(raw_values, 5)
                    supervised_values = timeseries_to_supervised(raw_values, lag = lag_days)
                    # normalize to 0 to 1
                    # supervised_values = supervised_values/supervised_values.max()
                    # normalize supervised_values
                    supervised_values -= np.mean(supervised_values, axis = 0) # zero-center
                    supervised_values /= np.std(supervised_values, axis = 0) # normalize
                        
                    # for input_features in ['pollution_val', 'one-hot-encoding+', 'glove-embedding+']:
                    for with_pollution_val in ['pollution_val', 'with_pol_val', 'without_pol_val']:
                        for input_features in ['one-hot+', 'one-hot+glove+']:
                            if with_pollution_val == 'pollution_val':
                                x_train_concat = supervised_values.copy()
                                input_features = ''
                            else:
                                X_concat_frames = pd.concat([X_train, X_valid, X_test])
                                feature_embeddings = generate_search_embedding(X_concat_frames, representation = 'one-hot')
                                feature_embeddings = lag_search_features(feature_embeddings, lag = search_lag)
                                if input_features == 'one-hot+':
                                    if with_pollution_val == 'with_pol_val':
                                        x_train_concat = np.concatenate((supervised_values, feature_embeddings), axis=1)
                                    else:
                                        x_train_concat = feature_embeddings.copy()
                                else:
                                    glove_feature_embeddings = generate_search_embedding(X_concat_frames, representation = 'glove')
                                    glove_feature_embeddings = lag_search_features(glove_feature_embeddings, lag = search_lag)
                                    if with_pollution_val == 'with_pol_val':
                                        x_train_concat = np.concatenate((supervised_values, feature_embeddings, glove_feature_embeddings), axis=1)
                                    else:
                                        x_train_concat = np.concatenate((feature_embeddings, glove_feature_embeddings), axis=1)

                            input_embedding = generate_input_sequence(x_train_concat, seq_length = seq_length)
                            input_embedding = input_embedding.reshape(len(input_embedding), -1)
                            # input_embedding -= np.mean(input_embedding, axis = 0) # zero-center
                            # input_embedding /= np.std(input_embedding, axis = 0) # normalize
                            y_class = [1 if i>pollution_value else 0 for i in raw_values]

                            # generate train_validation set 
                            x_train_valid = input_embedding[:train_len+valid_len]
                            y_train_valid = y_class[:train_len+valid_len]

                            x_test = input_embedding[train_len+valid_len:]
                            y_test = y_class[train_len+valid_len:]

                            valid_index = [i for i in range(train_len, train_len+valid_len)]
                            test_fold = [ -1 if i not in valid_index else 0 for i in range(0, len(x_train_valid)) ]
                            ps = PredefinedSplit(test_fold=test_fold)

                            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-5],
                                                 'C': [1, 10, 100, 1000]},
                                                {'kernel': ['linear'], 'C': [1, 10, 50]}]

                            clf = GridSearchCV(SVC(class_weight="balanced"), tuned_parameters, cv=ps,
                                               scoring='f1', verbose=0, n_jobs=2)

                            clf.fit(x_train_valid, y_train_valid)
                            pred = clf.predict(x_test)
                            accuracy = accuracy_score(y_test,pred)
                            f1_value = f1_score(y_test,pred)
                            y_score = clf.decision_function(x_test)
                            fpr,tpr,threshold = roc_curve(y_test, y_score) 
                            auc_value = auc(fpr,tpr) 

                            sheet1.write(row_index, col_index, with_pollution_val + '+' + input_features)
                            col_index = col_index + 1
                            sheet1.write(row_index, col_index, str(accuracy))
                            col_index = col_index + 1
                            sheet1.write(row_index, col_index,  str(f1_value))
                            col_index = col_index + 1
                            sheet1.write(row_index, col_index,  str(auc_value))
                            col_index = 0
                            row_index = row_index + 1

                            if with_pollution_val == 'pollution_val':
                                break
                            # fo.write(input_features + ',' + str(accuracy) +',' + str(f1_value) + ',' + str(auc_value)+ '\n')
        ws.save(file_out)
        del ws
    # book.save(file_out)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot ROC-AUC of air pollution prediction.')
    # Required file path
    parser.add_argument('-f','--file', type=str,
                    help='Path to the air pollution data')
    parser.add_argument('-fo','--file_out', type=str,
                    help='Path to the CSV stat data')
    args = parser.parse_args()
    if args.file:
        main(args.file, args.file_out)
    else:
        print("Input file path:-f")
