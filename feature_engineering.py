import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import datetime

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from settings import START_DATE, V_GROUPS_BY_NOTNULL


def generate_uid_features(df):

    df['bank_type'] = df['card3'].astype(str) + '_' + df['card5'].astype(str)
    df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    df['uid2'] = df['uid'].astype(str) + '_' + df['card3'].astype(str) + '_' + df['card5'].astype(str)
    df['uid3'] = df['uid2'].astype(str) + '_' + df['addr1'].astype(str) + '_' + df['addr2'].astype(str)
    df['uid4'] = df['uid3'].astype(str) + '_' + df['P_emaildomain'].astype(str)
    df['uid5'] = df['uid3'].astype(str) + '_' + df['R_emaildomain'].astype(str)

    return df


def exchange_rate_took_place_feature(df):
    df['is_foreign'] = df['TransactionAmt'].astype('str')\
        .str\
        .split('.', expand=True)[1]\
        .apply(lambda x: len(x) > 2)
    return df


def calc_smooth_mean(
        df,
        columns_agg,
        funcs=['mean', 'std', np.nanmedian],
        on='isFraud',
        m=50,
        to_round=5
):

    to_smooth = {}
    for f in funcs:

        f_name = f
        if not isinstance(f, str):
            f_name = f.__name__

        to_smooth[f_name] = df[on].agg(f_name)

    agg = df.groupby(columns_agg)[on].agg(['count'] + funcs)
    counts = agg['count']

    value_mappings = {}

    for f in funcs:

        f_name = f
        if not isinstance(f, str):
            f_name = f.__name__

        values = agg[f_name]
        smooth = (counts * values + m * to_smooth[f_name]) / (counts + m)

        smooth = smooth.to_dict()
        smooth_fixed = {}
        for k, v in smooth.items():
            k_fixed = [k] if not isinstance(k, tuple) else k
            smooth_fixed['_'.join(map(str, k_fixed))] = np.round(smooth[k], to_round)

        value_mappings[f_name] = smooth_fixed

    return value_mappings


def smoothed_encodings(
        df,
        columns_agg,
        column_value,
        funcs=['mean', 'std', np.nanmedian],
        train_size=None,
        m=50
):
    if train_size is None:
        train_size = df.shape[0]

    for col_agg in columns_agg:
        encoders = calc_smooth_mean(
            df.iloc[:int(train_size)],
            col_agg,
            funcs=funcs,
            on=column_value,
            m=m
        )

        for f in funcs:

            f_name = f
            if not isinstance(f, str):
                f_name = f.__name__

            col_name = '_'.join(col_agg)
            col_name = f'smoothed_encoded_{col_name}_on_{column_value}_{f_name}'

            df[col_name] = df[col_agg[0]].astype(str)
            for i in col_agg[1:]:
                df[col_name] += '_'
                df[col_name] += df[i].astype(str)

            df[col_name] = df[col_name].map(encoders[f_name])

    return df


def count_features(
        df,
        columns_agg,
        with_typical_for_user=True
):
    for c_agg in columns_agg:

        col_name = '_'.join(c_agg)

        df[f'{col_name}_count'] = df[c_agg + ['TransactionDT']]\
            .groupby(c_agg)['TransactionDT']\
            .transform('count')

        if with_typical_for_user and len(c_agg) > 1:
            df[f'{col_name}_count_how_typical'] = df[f'{col_name}_count'] / \
                df.groupby(['card1'])['TransactionDT'].transform('count')

    return df


def emaildomain_features(df):
    cols_P = ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']
    cols_R = ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

    df[cols_P] = df['P_emaildomain'].str.split('.', expand=True)
    df[cols_R] = df['R_emaildomain'].str.split('.', expand=True)

    for i in cols_P + cols_R:
        df[i] = df[i].astype(str)

    df['R=P'] = df['P_emaildomain'] == df['R_emaildomain']
    df['R1=P1'] = df['P_emaildomain_1'] == df['R_emaildomain_1']

    return df


def base_transaction_delta_features(df):
    df['time_from_prev_transaction'] = df \
        .groupby('card1')['TransactionDT'] \
        .diff()
    df['time_to_next_transaction'] = df \
        .groupby('card1')['time_from_prev_transaction'] \
        .shift(-1)

    return df


def add_datetime_features(df):

    tr_dt = 'TransactionDT'

    df[f'{tr_dt}_to_datetime'] = df[tr_dt].apply(
        lambda x: START_DATE + datetime.timedelta(seconds=x)
    )

    df[f'{tr_dt}_year'] = df[f'{tr_dt}_to_datetime'].dt.year - 2017
    df[f'{tr_dt}_month'] = df[f'{tr_dt}_to_datetime'].dt.month
    df[f'{tr_dt}_dayOfMonth'] = df[f'{tr_dt}_to_datetime'].dt.day
    df[f'{tr_dt}_dayOfWeek'] = df[f'{tr_dt}_to_datetime'].apply(
        lambda x: x.weekday()
    )
    df[f'{tr_dt}_weekOfMonth'] = (df[f'{tr_dt}_to_datetime'].dt.day - 1) // 7 + 1
    df[f'{tr_dt}_hour'] = df[f'{tr_dt}_to_datetime'].dt.hour
    #df[f'{tr_dt}_minute'] = df[f'{tr_dt}_to_datetime'].dt.minute
    #df[f'{tr_dt}_second'] = df[f'{tr_dt}_to_datetime'].dt.second
    df[f'{tr_dt}_split'] = (df[f'{tr_dt}_to_datetime'].dt.year - 2017) * 12 + \
                                df[f'{tr_dt}_to_datetime'].dt.month


    # US HOLIDAYS
    dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
    us_holidays = calendar().holidays(
        start=dates_range.min(),
        end=dates_range.max()
    )
    df['is_holiday'] = df[f'{tr_dt}_to_datetime']\
        .dt\
        .date\
        .astype('datetime64')\
        .isin(us_holidays)\
        .astype(np.int8)

    df['mean_time_between_transactions'] = df\
        .groupby('card1')['time_from_prev_transaction'] \
        .transform('mean')
    df['median_time_between_transactions'] = df \
        .groupby('card1')['time_from_prev_transaction'] \
        .transform(np.nanmedian)

    df['time_from_prev_transaction_ratio_to_mean'] = \
        df['time_from_prev_transaction'] / df['mean_time_between_transactions']

    df['time_from_prev_transaction_ratio_to_median'] = \
        df['time_from_prev_transaction'] / df['median_time_between_transactions']

    df['time_to_next_transaction_ratio_to_mean'] = \
        df['time_to_next_transaction'] / df['mean_time_between_transactions']

    df['time_to_next_transaction_ratio_to_median'] = \
        df['time_to_next_transaction'] / df[
            'median_time_between_transactions']

    df['Transaction_Number'] = df.groupby('card1').cumcount() + 1
    df['Transaction_Number_normed'] = df['Transaction_Number'] / df\
        .groupby('card1')['TransactionDT']\
        .transform('count')

    df.drop(
        labels=[f'{tr_dt}_year', f'{tr_dt}_to_datetime', f'{tr_dt}_month'],
        axis=1,
        inplace=True
    )

    return df


def encode_categorical_features(df, cat_features):
    encoders = {}

    for f in cat_features:
        le = preprocessing.LabelEncoder()

        df[f] = df[f].astype(str)

        df[f] = le.fit_transform(df[f].fillna('FILL_NAN'))
        encoders[f] = le

    return df, encoders


def process_id_33(df):
    df[['id_33_height', 'id_33_width']] = df['id_33'].str.split('x',
                                                                expand=True)

    df['id_33_width'] = df['id_33_width'].astype(float)
    df['id_33_height'] = df['id_33_height'].astype(float)

    return df


def process_id_30(df):
    id_30 = 'id_30'
    df['OS_NAME'] = df[id_30]
    df['OS_V0'] = df[id_30]
    df['OS_V1'] = df[id_30]
    df['OS_V2'] = df[id_30]

    def parse_version(string):

        versions = [None, None, None]
        sep = '.' if '.' in string else '_'

        if sep not in string:
            return versions
        else:
            splitted = string.split(sep)

            for ind, s in enumerate(splitted):
                versions[ind] = int(s)

            return versions

    df['OS_NAME'][df[id_30].notnull()] = df[df[id_30].notnull()][id_30].apply(
        lambda x: x if ' ' not in x else ' '.join(x.split()[:-1])
    )

    df['OS_V0'][df[id_30].notnull()] = df[df[id_30].notnull()][id_30].apply(
        lambda x: parse_version(x.split()[-1])[0]
    )

    df['OS_V1'][df[id_30].notnull()] = df[df[id_30].notnull()][id_30].apply(
        lambda x: parse_version(x.split()[-1])[1]
    )

    df['OS_V2'][df[id_30].notnull()] = df[df[id_30].notnull()][id_30].apply(
        lambda x: parse_version(x.split()[-1])[2]
    )

    df['OS_V_COMBINED'] = df['OS_V0'] * 1000 + df['OS_V1'] * 10 + df['OS_V2'].fillna(0)
    df['OS_V_COMBINED'] = df['OS_V_COMBINED'].astype(float)

    df['OS_V_MAJOR'] = df['OS_NAME'].astype(str) + '_' + df['OS_V0'].astype(str)

    return df


def features_to_PCA(df, columns, prefix, n_components=0.99):

    pca_reducer = PCA(n_components=n_components)

    transformed_values = pca_reducer.fit_transform(
        StandardScaler().fit_transform(
            df[columns].fillna(-0.5)
        )
    )

    size = transformed_values.shape[1]

    pca_df = pd.DataFrame(
        data=transformed_values,
        columns=[f'PCA_{prefix}{i}' for i in range(size)],
        index=df.index
    )

    #df.drop(labels=columns, axis=1, inplace=True)

    df = pd.concat([df, pca_df], axis=1)

    return df


def V_features_to_PCA(df):

    for k, v in V_GROUPS_BY_NOTNULL.items():
        df = features_to_PCA(df, v, prefix=f'V_{k}_GROUP_')

    return df


def C_features_to_PCA(df):
    return features_to_PCA(df, [f'C{i}' for i in range(1, 15)], prefix='C')


def D_features_to_PCA(df):
    return features_to_PCA(df, [f'D{i}' for i in range(1, 16)], prefix='D')


def log_features(df, colums):
    df[colums] = np.log(df[colums] + 1.0)
    return df


def C_log_features(df):
    return log_features(df, [f'C{i}' for i in range(1, 15)])


def add_is_null_features(
        df,
        columns=(
            'addr1',
            'addr2',
            'dist1',
            'dist2',
            'D1',
            'D2',
            'D3',
            'D5',
            'D6',
            'D7',
            'D8',
            'D9',
            'D10',
            'D11',
            'D12',
            'D13',
            'D14',
            'D15',
            'M1',  # == M2=3
            'M4',
            'M6',
            'M7',  # == M8-9
            'V1',  # == V1-11
            'V12',  # == V12-34
            'V35',  # V35-52
            'V53',  # V53-74
            'V75',  # V75-94
            'V95',  # V95-137
            'V138',  # V138-166
            'V167',
            'V322',
            'id_01',
            'id_02',
            'id_03',
            'id_04',
            'id_05',
            'id_06',
            'id_07',
            'id_08',
            'id_09',
            'id_10',
            'id_11',
            'id_12',
            'id_13',
            'id_14',
            'id_15',
            'id_16',
            'id_17',
            'id_18',
            'id_19',
            'id_20',
            'id_21',
            'id_22',
            'id_23',
            'id_24',
            'id_25',
            'id_26',
            'id_27',
            'id_28',
            'id_29',
            'id_30',
            'id_31',
            'id_32',
            'id_33',
            'id_34',
            'id_35',
            'id_36',
            'id_37',
            'id_38',
            'DeviceType',
            'DeviceInfo'
        )
):

    for column in columns:
        df[f'{column}_isnull'] = df[column].isnull()

    return df


def device_to_group(df):
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()

    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]

    df.loc[df['device_name'].str.contains('SM',
                                          na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG',
                                          na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-',
                                          na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G',
                                          na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto',
                                          na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto',
                                          na=False), 'device_name'] = 'Motorola'
    df.loc[
        df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[
        df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI',
                                          na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-',
                                          na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L',
                                          na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade',
                                          na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE',
                                          na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux',
                                          na=False), 'device_name'] = 'Linux'
    df.loc[
        df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[
        df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS',
                                          na=False), 'device_name'] = 'Asus'

    df.loc[df.device_name.isin(df.device_name.value_counts()[
                                   df.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    return df


def remove_rare_values(df, train_size, columns):
    for column in columns:

        valid_values = df[:train_size][column].value_counts()
        valid_values = valid_values[valid_values > 2]
        valid_values = set(valid_values.index)

        df[column] = np.where(
            df[column].isin(df[:train_size][column]),
            df[column],
            np.nan
        )

        df[column] = np.where(
            df[column].isin(valid_values),
            df[column],
            np.nan
        )

        return df