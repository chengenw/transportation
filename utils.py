import argparse
import random

import torch
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sdv.metadata import SingleTableMetadata
import os
import psutil

def check_mem():
    memory_info = psutil.virtual_memory()
    print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Memory Usage Percentage: {memory_info.percent}%")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', default=['GaussianCopula', 'CTGAN', 'TVAE', 'CTABGAN', 'STaSy', 'TabDDPM'], help='generative models')
    parser.add_argument('--dataset', type=str, default='green', help='dataset name: green, yellow, zone')
    parser.add_argument('--path', type=str, default='./dataset', help='dataset path')
    parser.add_argument('--dwn_model', type=str, default='GBM', help='downstream task model: LR, GBM or SVM')
    parser.add_argument('--dt_new_format', action='store_false', help='')
    parser.add_argument('--use_synthesizer', action='store_false', help='')
    parser.add_argument('--viz', action='store_true', help='')
    parser.add_argument('--single_col', action='store_true', help='')
    parser.add_argument('--compare_all', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--nums', type=int, nargs='+', default=[1000, 10000, 40000, 100000, 400000], help='nums of data row')
    parser.add_argument('--nexp', type=int, default=3, help='nums of trained generative models')
    parser.add_argument('--ngen', type=int, default=5, help='nums of generation per model')
    parser.add_argument('--epochs', type=int, default=300, help='nums of epochs'   )
    parser.add_argument('--outfile', type=str, default='result')
    parser.add_argument('--col_name', type=str, default='total_amount', help=['tip_amount', 'total_amount', 'dropoff_time', 'dropoff_longitude', 'trip_distance', 'fare_amount'])
    parser.add_argument('--train_base_info', action='store_true', help='')
    parser.add_argument('--hierarchical', action='store_true', help='')
    parser.add_argument('--method_h', default='HCTGAN', help='')
    parser.add_argument('--verbose', action='store_false', help='')
    parser.add_argument('--metric_base_info', action='store_true', help='')
    parser.add_argument('--metric_extra_info', action='store_true', help='')
    parser.add_argument('--more_cols', action='store_false', help='')
    parser.add_argument('--syn2', action='store_true', help='')
    parser.add_argument('--save_model', action='store_true', help='')
    parser.add_argument('--load_model', action='store_true', help='')
    parser.add_argument('--d_steps', type=int, default=1, help='Discriminator steps')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='Discriminator learning rate')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='Generator learning rate')
    parser.add_argument('--max_clusters', type=int, default=10, help='max number of clusters in ClusterBasedNormalizer')
    parser.add_argument('--row_aligned', action='store_false', help='')
    parser.add_argument('--decoder_only', action='store_true', help='')
    parser.add_argument('--base_less', action='store_true', help='')
    parser.add_argument('--base_more', action='store_true', help='')
    # parser.add_argument('--no_dwn', action='store_true', help='downstream task')
    parser.add_argument('--no_sdv', action='store_true', help='sdv score')
    parser.add_argument('--no_wd', action='store_true', help='ignore wasserstein distance test')
    parser.add_argument('--no_cov', action='store_true', help='ignore coverage test')
    parser.add_argument('--no_prv', action='store_true', help='ignore prv test')
    parser.add_argument('--fake_cond', action='store_true', help='use samples from the base generator as condition')
    parser.add_argument('--dwn_base', action='store_true', help='downstream tasks only use base info + col_name')
    parser.add_argument('--statCond', action='store_true', help='')
    parser.add_argument('--percs', type=float, nargs='+', default=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], help='percentile for dcr_rs, dcr_hs')

    args = parser.parse_args()
    assert not args.metric_base_info or args.metric_base_info and not args.train_base_info
    assert not args.hierarchical or args.hierarchical and not args.train_base_info and not args.metric_base_info
    assert not (args.dwn_base and args.metric_base_info)

    return args

def get_metadata(taxi_data, hierarchical=False):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=taxi_data)
    metadata.primary_key = None
    cols_numerical = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_datetime', 'dropoff_datetime']
    cols_categorical = ['pulocationid', 'dolocationid']
    if not hierarchical:
        for col in cols_numerical:
            if col in metadata.columns:
                metadata.update_column(column_name=col, sdtype='numerical')
        for col in cols_categorical:
            if col in metadata.columns:
                metadata.update_column(column_name=col, sdtype='categorical')

    return metadata

def get_metadata_CA(taxi_data, cat_idx):
    columns = taxi_data.columns
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=taxi_data)
    metadata.primary_key = None

    for i, col in enumerate(columns):
        if i in cat_idx:
            metadata.update_column(column_name=col, sdtype='categorical')
        else:
            metadata.update_column(column_name=col, sdtype='numerical')

    return metadata

def cat_metadata(metadata):
    d_metadata = metadata.to_dict()
    cat_col = []
    num_col = []
    general_col = []
    for col in d_metadata['columns'].keys():
        if d_metadata['columns'][col]['sdtype'] == 'categorical':
            cat_col.append(col)
        elif d_metadata['columns'][col]['sdtype'] == 'numerical':
            num_col.append(col)
        else:
            general_col.append(col)
    return cat_col, num_col, general_col

def get_col_idx(columns):
    cat_col, cat_idx = [], []
    int_col, int_idx = [], []
    col_cat = ['pickup_weekday', 'dropoff_weekday', 'vendorid', 'rate_code', 'passenger_count', 'payment_type', 'pulocationid', 'dolocationid']
    col_int = ['pickup_time', 'dropoff_time']
    for i, col in enumerate(columns):
        if col in col_cat:
            cat_col.append(col)
            cat_idx.append(i)
        elif col in col_int:
            int_col.append(col)
            int_idx.append(i)

    return cat_idx, int_idx

def to_idx_col(pd_data):
    columns = [str(i) for i in range(pd_data.shape[1])]
    data = pd.DataFrame(pd_data.to_numpy().astype('float'), columns=columns)
    return data

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def dummify(X, cat_indexes, divide_by=0, drop_first=False):
    df = pd.DataFrame(X, columns = [str(i) for i in range(X.shape[1])]) # to Pandas
    df_names_before = df.columns
    for i in cat_indexes:
        df = pd.get_dummies(df, columns=[str(i)], prefix=str(i), dtype='float', drop_first=drop_first)
        if divide_by > 0: # needed for L1 distance to equal 1 when categories are different
            filter_col = [col for col in df if col.startswith(str(i) + '_')]
            df[filter_col] = df[filter_col] / divide_by
    df_names_after = df.columns
    df = df.to_numpy()
    return df, df_names_before, df_names_after

# Mixed data is tricky, nearest neighboors (for the coverage) and Wasserstein distance (based on L2) are not scale invariant
# To ensure that the scaling between variables is relatively uniformized, we take inspiration from the Gower distance used in mixed-data KNNs: https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b
# Continuous: we do min-max normalization (to use Gower |x1-x2|/(max-min) as distance)
# Categorical: We one-hot and then divide by 2 (e.g., 0 0 0.5 with 0.5 0 0 will have distance 0.5 + 0.5 = 1)
# After these transformations, taking the L1 (City-block / Manhattan distance) norm distance will give the Gower distance
def minmax_scale_dummy(X_train, X_test, cat_indexes=[], mask=None, divide_by=2):
    X_train_ = copy.deepcopy(X_train)
    X_test_ = copy.deepcopy(X_test)
    # normalization of continuous variables
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    if len(cat_indexes) != X_train_.shape[1]: # if all variables are categorical, we do not scale-transform
        not_cat_indexes = [i for i in range(X_train_.shape[1]) if i not in cat_indexes]
        scaler.fit(X_train_[:, not_cat_indexes])

        #Transforms
        X_train_[:, not_cat_indexes] = scaler.transform(X_train_[:, not_cat_indexes])
        X_test_[:, not_cat_indexes] = scaler.transform(X_test_[:, not_cat_indexes])

    # One-hot the categorical variables (>=3 categories)
    df_names_before, df_names_after = None, None
    n = X_train.shape[0]
    if len(cat_indexes) > 0:
        X_train_test, df_names_before, df_names_after = dummify(np.concatenate((X_train_, X_test_), axis=0), cat_indexes, divide_by=divide_by)
        X_train_ = X_train_test[0:n,:]
        X_test_ = X_train_test[n:,:]

    # 1 2 3 4 6 4_1 4_2 7_1 7_2 7_3

    # We get the new mask now that there are one-hot features
    if mask is not None:
        if len(cat_indexes) == 0:
            return X_train_, X_test_, mask, scaler, df_names_before, df_names_after
        else:
            mask_new = np.zeros(X_train_.shape)
            for i, var_name in enumerate(df_names_after):
                if '_' in var_name: # one-hot
                    var_ind = int(var_name.split('_')[0])
                else:
                    var_ind = int(var_name)
                mask_new[:, i] = mask[:, var_ind]
            return X_train_, X_test_, mask_new, scaler, df_names_before, df_names_after
    else:
        return X_train_, X_test_, scaler, df_names_before, df_names_after

def to_graph(dataset, args):
    if args.dataset != 'zone':
        return {}

    # dataset with two columns (orig, dest)
    d = {}
    for _, row in dataset.iterrows():
        k = (int(row['pulocationid']), int(row['dolocationid']))
        if k in d:
            d[k] += 1
        else:
            d[k] = 1

    num = len(dataset)
    for k, v in d.items():
        d[k] = v / num

    return d

def diff_graph(graph, graph2, args): # total variation distance
    if args.dataset != 'zone':
        raise NotImplementedError

    v_diff = 0

    for k, v in graph.items():
        if k in graph2:
            if graph2[k] < v:
                v_diff += v - graph2[k]
        else:
            v_diff += v

    for k, v in graph2.items():
        if k in graph:
            if graph[k] < v:
                v_diff += v - graph[k]
        else:
            v_diff += v

    v_diff = v_diff / 2 # normalizing; range [0, 1]
    return v_diff

def read_pure_data(path, split='train'):
    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)
    X_num = None
    X_cat = None
    if os.path.exists(os.path.join(path, f'X_num_{split}.npy')):
        X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')):
        X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True)

    return X_num, X_cat, y

def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)

def make_dataset(
        data_path: str,
        num_classes: int,
        is_y_cond: bool
):
    # classification
    if num_classes > 0:
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) or not is_y_cond else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if not is_y_cond:
                X_cat_t = concat_y_to_X(X_cat_t, y_t)
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) or not is_y_cond else None
        y = {}

        for split in ['train', 'val', 'test']:
            X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
            if not is_y_cond:
                X_num_t = concat_y_to_X(X_num_t, y_t)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            y[split] = y_t

    if X_cat:
        X_cat = np.concatenate((X_cat['train'], X_cat['val'], X_cat['test']), axis=0)
    else:
        X_cat = None
    if X_num:
        X_num = np.concatenate((X_num['train'], X_num['val'], X_num['test']), axis=0)
    if X_cat is None:
        X = X_num
    elif X_num is None:
        X = X_cat
    else:
        X = np.concatenate((X_cat, X_num), axis=1)
    cat_idx = [] if X_cat is None else [i for i in range(X_cat.shape[1])]

    return X, cat_idx
