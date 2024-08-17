import os
import glob
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def import_ts_data_unsupervised(data_root, data, entities=None, combine=False):
    if type(entities) == str:
        entities_lst = entities.split(',')
    elif type(entities) == list:
        entities_lst = entities
    else:
        raise ValueError('wrong entities')

    name_lst = []
    train_lst = []
    test_lst = []
    label_lst = []

    if len(glob.glob(os.path.join(data_root, data) + '/*.csv')) == 0:
        if data == 'DSADS':
            machine_lst = os.listdir(data_root + data + '/')
            for m in sorted(machine_lst):
                if entities != 'FULL' and m not in entities_lst:
                    continue
                train_path = glob.glob(os.path.join(data_root, data, m, '*train*.csv'))
                test_path = glob.glob(os.path.join(data_root, data, m, '*test*.csv'))

                assert len(train_path) == 1 and len(test_path) == 1, f'{m}'
                train_path, test_path = train_path[0], test_path[0]

                scaler = StandardScaler()

                train_df = pd.read_csv(train_path, sep=',', index_col=0)
                test_df = pd.read_csv(test_path, sep=',', index_col=0)
                labels = test_df['label'].values
                train_df, test_df = train_df.drop('label', axis=1), test_df.drop('label', axis=1)

                # normalization
                scaler.fit(train_df)
                train = scaler.transform(train_df)
                test = scaler.transform(test_df)

                train_lst.append(train)
                test_lst.append(test)
                label_lst.append(labels)
                name_lst.append(m)

            if combine:
                train_lst = [np.concatenate(train_lst)]
                test_lst = [np.concatenate(test_lst)]
                label_lst = [np.concatenate(label_lst)]
                name_lst = [data + '_combined']

        else:
            # DCdetector预处理
            scaler = StandardScaler()
            df = np.load(f'{data_root}{data}/{data}_train.npy')
            scaler.fit(df)
            df = scaler.transform(df)
            test_data = np.load(f'{data_root}{data}/{data}_test.npy')
            test = scaler.transform(test_data)
            train = df
            labels = np.load(f'{data_root}{data}/{data}_test_label.npy')
            label = np.squeeze(labels)
            train_lst.append(train)
            test_lst.append(test)
            label_lst.append(label)
            name_lst.append(data)

    else:
        if data == 'Epilepsy':
            scaler = StandardScaler()
            df = pd.read_csv(f'{data_root}{data}/{data}_train.csv', sep=',', index_col=0)
            test_data = pd.read_csv(f'{data_root}{data}/{data}_test.csv', sep=',', index_col=0)
            labels = test_data['label'].values
            df, test_data = df.drop('label', axis=1), test_data.drop('label', axis=1)
            df = np.nan_to_num(df)
            scaler.fit(df)
            df = scaler.transform(df)
            test_data = np.nan_to_num(test_data)
            test = scaler.transform(test_data)
            train = df
            label = np.squeeze(labels)
        else:
            scaler = StandardScaler()
            df = pd.read_csv(f'{data_root}{data}/{data}_train.csv')
            df = df.values[:, 1:]
            df = np.nan_to_num(df)
            scaler.fit(df)
            df = scaler.transform(df)
            test_data = pd.read_csv(f'{data_root}{data}/{data}_test.csv')
            test_data = test_data.values[:, 1:]
            test_data = np.nan_to_num(test_data)
            test = scaler.transform(test_data)
            train = df
            labels = pd.read_csv(f'{data_root}{data}/{data}_test_label.csv').values[:, 1:]
            label = np.squeeze(labels)

        train_lst.append(train)
        test_lst.append(test)
        label_lst.append(label)
        name_lst.append(data)

    return train_lst, test_lst, label_lst, name_lst


