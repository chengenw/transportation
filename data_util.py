import pandas as pd
from sklearn.datasets import load_iris, load_wine, fetch_california_housing
import os
import wget
import matplotlib.pyplot as plt
import seaborn as sns

def load_green(path, num, like_yellow_data=False):
    file = f'{path}/2015_green_{num}.csv'
    taxi_data = pd.read_csv(file, parse_dates=['pickup_datetime', 'dropoff_datetime'],
                            index_col=0)  # unable to detect datetime??
    taxi_data = taxi_data.drop(['Ehail_fee'], axis=1)  # due to 'NaN'
    if like_yellow_data:
        taxi_data = taxi_data.drop(['Trip_type', 'Improvement_surcharge'], axis=1)

    return taxi_data

def load_yellow(path, num): # not fully tested
    file = f'{path}/2015_yellow_{num}.csv'
    taxi_data = pd.read_csv(file, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], index_col=0)
    taxi_data.rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'tpep_dropoff_datetime': 'dropoff_datetime',
                              'RateCodeID': 'rate_code'}, inplace=True)

    return taxi_data

def preprocess_tab(taxi_data):
    taxi_data.columns = map(str.lower, taxi_data.columns)
    taxi_data['store_and_fwd_flag'] = (taxi_data['store_and_fwd_flag'] == 'Y').astype(int)

    taxi_data = taxi_data.dropna()  # remove missing values

    taxi_data['pickup_weekday'] = taxi_data['pickup_datetime'].dt.weekday
    taxi_data['dropoff_weekday'] = taxi_data['dropoff_datetime'].dt.weekday
    taxi_data['pickup_time'] = taxi_data['pickup_datetime'].dt.time
    taxi_data['dropoff_time'] = taxi_data['dropoff_datetime'].dt.time
    taxi_data['pickup_time'] = taxi_data['pickup_time'].apply(
        lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    taxi_data['dropoff_time'] = taxi_data['dropoff_time'].apply(
        lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    taxi_data = taxi_data.drop(['pickup_datetime', 'dropoff_datetime'], axis=1)
    return taxi_data

def get_idx(columns):
    cat_idx = []
    int_idx = []
    col_cat = ['pickup_weekday', 'dropoff_weekday', 'vendorid', 'rate_code', 'passenger_count', 'payment_type', 'pulocationid', 'dolocationid', 'trip_type', 'store_and_fwd_flag']
    col_int = ['pickup_time', 'dropoff_time']
    for i, col in enumerate(columns):
        if col in col_cat:
            cat_idx.append(i)
        elif col in col_int:
            int_idx.append(i)

    return cat_idx, int_idx

def load_zone(path, num, seed):
    sourceFile = f'{path}/tracrData/green_tripdata_2019-03.parquet'
    taxi_data = pd.read_parquet(sourceFile, engine='pyarrow')
    taxi_data = taxi_data.sample(n=num, random_state=seed)
    taxi_data.rename(columns={"RateCodeID":'rate_code','lpep_pickup_datetime':'pickup_datetime', 'lpep_dropoff_datetime':'dropoff_datetime'}, inplace=True)
    taxi_data = taxi_data.drop('ehail_fee', axis=1)

    print(taxi_data.columns)

    return taxi_data

def fetch_qsar_biodegradation():
    path = 'data/qsar_biodegradation'
    if not os.path.isdir(path):
        os.mkdir(path)
    file = f'{path}/biodeg.csv'
    if not os.path.exists(file):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out=path)

    with open(file, 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header = None)
        X = df.iloc[:, :-1].astype('float')
        X.columns = [str(i) for i in range(X.shape[1])]
        y = pd.factorize(df.iloc[:,-1])[0] # str to numeric
        y = pd.Series(y)

    return  X, y

def data_loader(dataset, y_column='target'):
    if dataset == 'california':
        X, y = fetch_california_housing(as_frame=True, return_X_y=True)
        Xy = pd.concat([X, y.rename(y_column)], axis=1)
        cat_idx = []
        int_idx = [1, 4]
        cat_y = False

    elif dataset == 'qsar_biodegradation':
        X, y = fetch_qsar_biodegradation()
        Xy = pd.concat([X, y.rename(y_column)], axis=1)
        int_idx = [2, 3, 4, 5, 6, 8, 9, 10, 15, 18, 19, 20, 22, 25, 31, 32, 33, 34, 37, 39, 40]
        cat_idx = [23, 24, 28]
        cat_y = True

    return Xy, cat_idx, int_idx, cat_y

def plot_speed_plt(file, num): # complexity plot
    df = pd.read_csv(file, skipinitialspace=True)
    df = df[df['num'] == num]
    stats = df.groupby('method')['minutes'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(range(len(df)))
    bars = plt.bar(stats['method'], stats['mean'], color=colors)
    # plt.yscale('log')
    plt.xlabel('Method')
    plt.ylabel('Minutes')
    # plt.title('Time to train the model')
    plt.savefig('output/speed_plt.pdf')
    plt.show()

def plot_speed(file):
    df = pd.read_csv(file, skipinitialspace=True)

    sns.barplot(data=df, x='method', y='minutes', hue='num')
    # plt.yscale('log')
    plt.savefig('output/speed.pdf')
    plt.show()

def plot_ratio(file, num): # rDCR plot
    df = pd.read_csv(file, skipinitialspace=True)
    df = df[df['num'] == num]
    hue = 'perc'
    df[hue] = df[hue].astype('category')
    ax = sns.barplot(df, x='method', y='rDCR', hue=hue)
    ax.axhline(y=1.0, color='orange', linestyle='--')
    plt.savefig('output/ratio.pdf')
    plt.title(f'DCR ratio, training on {num} samples')
    plt.show()

if __name__ == '__main__':
    f_speed = 'output/speed_.csv'
    f_rDCR = 'output/dcr_.csv'
    # plot_speed_plt('output/method_speed.csv', 40000)
    # plot_speed('output/method_speed.csv')
    plot_speed_plt(f'{f_speed}', 40000)
    # plot_speed(f'{f_speed}')

    # plot_ratio('output/dcr.csv', 80000)
    # plot_ratio('output/dcr.csv', 180000)
    plot_ratio(f'{f_rDCR}', 40000)
    # plot_ratio(f'{f_rDCR}', 180000)
