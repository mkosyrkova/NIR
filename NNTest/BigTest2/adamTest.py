from sklearn.neural_network import MLPRegressor
import sklearn
import numpy as np
import _pickle as pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
import pandas

input_files = ['preproc.pcl', 'cut_preproc.pcl']
# input_files = ['preproc.pcl', 'preproc_with_discr_minmax.pcl', 'preproc_with_title_minmax.pcl',
#                'cut_preproc.pcl', 'cut_preproc_with_discr_minmax.pcl',
#                'cut_preproc_with_title_minmax.pcl']
min_max_scaler = preprocessing.MinMaxScaler()
layers1 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
learning_rate_inits = [0.001, 0.0001]

def load(file_path):
    start = time.time()
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print('Read from .pcl time: {} sec'.format(time.time() - start))
    return data


def dump(file_path_dump, file):
    with open(file_path_dump, 'wb') as f:
        pickle.dump(file, f)


for input_file in input_files:
    time_to_fit = list()

    loss_train_list = list()
    loss_test_list = list()
    loss_train_minmax_list = list()
    loss_test_minmax_list = list()

    layer1_list = list()
    layer2_list = list()
    learning_rate_init_list =list()
    solver_list = list()
    activation_list = list()
    max_iter_list = list()
    model_name_list = list()
    data = load(input_file)
    y = data['deal_probability'].ravel()
    x = data.drop(columns=['deal_probability', 'item_id', 'user_id', 'image', 'activation_date', 'standart_price',
                           'standart_item_seq_number', 'standart_image_top_1', 'standart_region', 'standart_city',
                           'standart_parent_category_name', 'standart_category_name', 'standart_param_1',
                           'standart_param_2',
                           'standart_param_3', 'standart_user_type', 'sigm_price',
                           'sigm_item_seq_number', 'sigm_image_top_1', 'sigm_region', 'sigm_city',
                           'sigm_parent_category_name',
                           'sigm_category_name', 'sigm_param_1', 'sigm_param_2', 'sigm_param_3', 'sigm_user_type',
                           'standart_min_max_price', 'standart_min_max_item_seq_number', 'standart_min_max_image_top_1',
                           'standart_min_max_region',
                           'standart_min_max_city', 'standart_min_max_parent_category_name',
                           'standart_min_max_category_name',
                           'standart_min_max_param_1', 'standart_min_max_param_2', 'standart_min_max_param_3',
                           'standart_min_max_user_type',
                           'standart_sigm_price', 'standart_sigm_item_seq_number', 'standart_sigm_image_top_1',
                           'standart_sigm_region',
                           'standart_sigm_city', 'standart_sigm_parent_category_name', 'standart_sigm_category_name',
                           'standart_sigm_param_1',
                           'standart_sigm_param_2', 'standart_sigm_param_3', 'standart_sigm_user_type'
                           ])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    for layer1 in layers1:
        for layer2 in range(1, layer1):
           for learning_rate_init in learning_rate_inits:
                    model_name = '{}_{}_{}_{}_{}'.format(layer1,layer2,'adam', learning_rate_init, 'relu')
                    layer1_list.append(layer1)
                    layer2_list.append(layer2)
                    solver_list.append('adam')
                    learning_rate_init_list.append(learning_rate_init)
                    max_iter_list.append(500)
                    model_name_list.append(model_name)
                    activation_list.append('relu')
                    model = MLPRegressor(
                                hidden_layer_sizes=(layer1,layer2), activation='relu', solver='adam',
                                learning_rate_init = learning_rate_init, max_iter=500,
                                random_state=9, early_stopping=True)
                    start = time.time()
                    model.fit(X_train, y_train)
                    time_to_fit.append(time.time() - start)
                    with open('D:/adamTest2/models/' + model_name +"_"+ input_file, 'wb') as f:
                         pickle.dump(model, f)

                    predict_test = model.predict(X_test)
                    predict_test_minmax = min_max_scaler.fit_transform(predict_test.reshape(-1, 1))
                    loss_test_list.append(sklearn.metrics.mean_squared_error(predict_test, y_test))  # mean sum squared loss
                    loss_test_minmax_list.append(sklearn.metrics.mean_squared_error(predict_test_minmax, y_test))  # mean sum squared loss
                    concat_test = np.column_stack((predict_test, y_test))
                    predict_train = model.predict(X_train)
                    predict_train_minmax = min_max_scaler.fit_transform(predict_train.reshape(-1, 1))
                    loss_train_list.append(sklearn.metrics.mean_squared_error(predict_train, y_train))  # mean sum squared loss
                    loss_train_minmax_list.append(sklearn.metrics.mean_squared_error(predict_train_minmax, y_train))  # mean sum squared loss
                    concat_train = np.column_stack((predict_train, y_train))

                    with open('D:/adamTest2/predicts_test/' + model_name +"_"+ input_file, 'wb') as f:
                        pickle.dump(concat_test, f)
                    # with open("time_and_loss_test.txt", 'a') as file:
                    #     file.write(model_name + " "+input_file + time_to_fit + time_to_predict + loss_test + '\n')
                    with open('D:/adamTest2/predicts_train/' + model_name + "_"+ input_file, 'wb') as f:
                         pickle.dump(concat_train, f)
                    # with open("time_and_loss_train.txt", 'a') as file:
                    #     file.write(model_name +" "+ input_file + loss_train + '\n')
        print (layer1)

    d = { 'solver': solver_list, 'layer1': layer1_list , 'layer2': layer2_list, 'activation': activation_list,'maxiter':max_iter_list,'learning_rate_init': learning_rate_init_list,'time_fit': time_to_fit,
          'loss_test': loss_test_list, 'loss_train': loss_train_list,'loss_test_minmax': loss_test_minmax_list, 'loss_train_minmax': loss_train_minmax_list, 'model_name': model_name_list}
    df = pandas.concat([pandas.Series(v, name=k) for k, v in d.items()], axis=1)
    dump("D:/adamTest2/time_and_loss_" + input_file, df)