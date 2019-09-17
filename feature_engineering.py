import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import datetime

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold


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

    df['cents'] = np.round(df['TransactionAmt'] - df['TransactionAmt'].astype(int), 3)
    df['dollars'] = df['TransactionAmt'].astype(int).astype(str)
    df['cents_categorical'] = df['cents'].astype(str)

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
        with_typical_for_user=True,
        remove_rare=False,
        threshold_rare=50,
        fill_value=-9999
):
    for c_agg in columns_agg:

        col_name = '_'.join(c_agg)

        df[f'{col_name}_count'] = df[c_agg + ['TransactionDT']]\
            .groupby(c_agg)['TransactionDT']\
            .transform('count')

        if with_typical_for_user and len(c_agg) > 1:
            df[f'{col_name}_count_how_typical_by_card1'] = df[f'{col_name}_count'] / \
                df.groupby(['card1'])['TransactionDT'].transform('count')

            df[f'{col_name}_count_how_typical_subcard'] = df[f'{col_name}_count'] / \
                                                  df.groupby(['subcard_categorical'])[
                                                      'TransactionDT'].transform(
                                                      'count')

        # we remove only single features
        if remove_rare and len(c_agg) == 1:
            df[f'{col_name}_removed_rare'] = df[col_name].copy()
            df[
                df[f'{col_name}_count'] < threshold_rare
            ][f'{col_name}_removed_rare'] = fill_value

    return df


def extract_registration_date(df):
    tr_dt = 'TransactionDT'

    df[f'{tr_dt}_to_datetime'] = df[tr_dt].apply(
        lambda x: START_DATE + datetime.timedelta(seconds=x)
    )

    df['card_registered_delta_tmp'] = pd.to_timedelta(df['D1'], unit='day')
    df['subcard_reg_date'] = (
            df['TransactionDT_to_datetime'] - df['card_registered_delta_tmp']
    )
    df['subcard_reg_timestamp'] = df['subcard_reg_date']\
        .dt\
        .date\
        .apply(
        lambda x: (
                x - datetime.date(1970, 1, 1)
        ).total_seconds()
    )
    df['subcard_categorical'] = df['card1'].astype(str) + '_' + df['subcard_reg_date']\
        .dt\
        .date\
        .astype(str)

    df.drop(
        labels=[
            'card_registered_delta_tmp',
            'subcard_reg_date'
        ],
        axis=1,
        inplace=True
    )

    return df


def emaildomain_features(df):
    cols_P = ['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']
    cols_R = ['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

    df[cols_P] = df['P_emaildomain'].str.split('.', expand=True)
    df[cols_R] = df['R_emaildomain'].str.split('.', expand=True)

    for i in cols_P + cols_R:
        df[i] = df[i].astype(str)

    # df['R=P'] = df['P_emaildomain'] == df['R_emaildomain']
    # df['R1=P1'] = df['P_emaildomain_1'] == df['R_emaildomain_1']

    return df


def base_transaction_delta_features(
        df,
        column_aggs=[
            ['card1', 'subcard_categorical'],
        ]
):

    for c_agg in column_aggs:

        col_name = '_'.join(c_agg)

        df[f'time_from_prev_transaction_by_{col_name}'] = df \
            .groupby(c_agg)['TransactionDT'] \
            .diff()
        df[f'time_to_next_transaction_by_{col_name}'] = df \
            .groupby(c_agg)[f'time_from_prev_transaction_by_{col_name}'] \
            .shift(-1)

    return df


def base_transaction_delta_features(
        df,
        column_aggs=[
            ['card1'],
            ['card1', 'ProductCD'],
            ['card1', 'ProductCD', 'addr1'],
            ['card1', 'subcard_categorical'],
        ]
):

    for c_agg in column_aggs:

        col_name = '_'.join(c_agg)

        df[f'time_from_prev_transaction_by_{col_name}'] = df \
            .groupby(c_agg)['TransactionDT'] \
            .diff()
        df[f'time_to_next_transaction_by_{col_name}'] = df \
            .groupby(c_agg)[f'time_from_prev_transaction_by_{col_name}'] \
            .shift(-1)

    return df


def add_datetime_features(df):

    tr_dt = 'TransactionDT'

    df[f'{tr_dt}_year'] = df[f'{tr_dt}_to_datetime'].dt.year - 2017
    df[f'{tr_dt}_month'] = df[f'{tr_dt}_to_datetime'].dt.month
    df[f'{tr_dt}_dayOfMonth'] = df[f'{tr_dt}_to_datetime'].dt.day
    # df[f'{tr_dt}_dayOfWeek'] = df[f'{tr_dt}_to_datetime'].apply(
    #     lambda x: x.weekday()
    # )
    # df[f'{tr_dt}_weekOfMonth'] = (df[f'{tr_dt}_to_datetime'].dt.day - 1) // 7 + 1
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

    for agg in [['card1', 'subcard_categorical']]:

        col_name = '_'.join(agg)

        df[f'{col_name}_mean_time_between_transactions'] = df\
            .groupby(agg)[f'time_from_prev_transaction_by_{col_name}'] \
            .transform('mean')
        df[f'{col_name}_median_time_between_transactions'] = df \
            .groupby(agg)[f'time_from_prev_transaction_by_{col_name}'] \
            .transform(np.nanmedian)
        df[f'time_from_prev_transaction_by_{col_name}_ratio_to_mean'] = \
            df[f'time_from_prev_transaction_by_{col_name}'] / \
            df[f'{col_name}_mean_time_between_transactions']

        df[f'time_from_prev_transaction_by_{col_name}_ratio_to_median'] = \
            df[f'time_from_prev_transaction_by_{col_name}'] / \
            df[f'{col_name}_median_time_between_transactions']

        df[f'time_to_next_transaction_by_{col_name}_ratio_to_mean'] = \
            df[f'time_to_next_transaction_by_{col_name}'] / \
            df[f'{col_name}_mean_time_between_transactions']

        df[f'time_to_next_transaction_by_{col_name}_ratio_to_median'] = \
            df[f'time_to_next_transaction_by_{col_name}'] / \
            df[f'{col_name}_median_time_between_transactions']

    df.reset_index(inplace=True)
    df.set_index('TransactionDT_to_datetime', inplace=True)

    for interval in ['1min', '10min']:
        df[f'TransactionAmt_count_within_{interval}'] = df\
            .groupby('card1')['TransactionAmt']\
            .rolling(interval)\
            .count()\
            .reset_index()\
            .sort_values('TransactionDT_to_datetime')['TransactionAmt']\
            .values

        df[f'TransactionAmt_sum_within_{interval}'] = df \
            .groupby('card1')['TransactionAmt'] \
            .rolling(interval) \
            .sum()\
            .reset_index()\
            .sort_values('TransactionDT_to_datetime')['TransactionAmt']\
            .values

        df[f'TransactionAmt_mean_within_{interval}'] = df \
            .groupby('card1')['TransactionAmt'] \
            .rolling(interval) \
            .mean()\
            .reset_index()\
            .sort_values('TransactionDT_to_datetime')['TransactionAmt']\
            .values

        df[f'TransactionAmt_std_within_{interval}'] = df \
            .groupby('card1')['TransactionAmt'] \
            .rolling(interval) \
            .std()\
            .reset_index()\
            .sort_values('TransactionDT_to_datetime')['TransactionAmt']\
            .values

        df[f'TransactionAmt_unique_within_{interval}'] = df \
            .groupby('card1')['TransactionAmt'] \
            .rolling(interval) \
            .apply(lambda x: len(np.unique(x))) \
            .reset_index() \
            .sort_values('TransactionDT_to_datetime')['TransactionAmt'] \
            .values

    df['Transaction_Number'] = df.groupby('subcard_categorical').cumcount() + 1
    df['Transaction_Number_normed'] = df['Transaction_Number'] / df\
        .groupby('subcard_categorical')['TransactionDT']\
        .transform('count')

    # df.drop(
    #     labels=[f'{tr_dt}_year', f'{tr_dt}_to_datetime', f'{tr_dt}_month'],
    #     axis=1,
    #     inplace=True
    # )

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
    df[['id_33_height', 'id_33_width']] = df['id_33'].str.split('x', expand=True)

    df['id_33_width'] = df['id_33_width'].astype(float)
    df['id_33_height'] = df['id_33_height'].astype(float)

    return df


def process_id_31(df):

    df['id_31_BROWSER_NAME'] = df['id_31'].fillna('NAN_BROWSER').apply(
        lambda x: ' '.join([i for i in x.split() if '.' not in i])
    )

    df['id_31_BROWSER_VERSION'] = df['id_31'].fillna('NAN_BROWSER').apply(
        lambda x: ' '.join([i for i in x.split()     if '.' in i])
    )

    df['id_31_BROWSER_VERSION'] = df['id_31_BROWSER_VERSION']\
        .replace(
        {
            'NAN_BROWSER': np.nan,
            '': np.nan
        }
    )

    df['id_31_BROWSER_VERSION'] = df['id_31_BROWSER_VERSION'].astype(float)

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
    df['OS_VERSION_FULL'] = df['OS_V0'].astype(str) + '_' + df['OS_V1'].astype(str)\
                            + '_' + df['OS_V2'].astype(str)

    df['OS_VERSION_MINIMAL'] = df['OS_V0'].astype(str) + '_' + df['OS_V1'].astype(str)

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
        columns
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


def V_groups_to_nan(df):
    for k, v in V_GROUPS_BY_NOTNULL.items():
        df[f'{k}_notnull'] = df[{v[0]}].notnull()

    return df


def advanced_V_processing(df):

    # df['V126-137_mean_with_zeros'] = df[[f'V{i}' for i in range(126, 138)]].mean(axis=1)
    # df['V126-137_std'] = df[[f'V{i}' for i in range(126, 138)]].mean(axis=1)
    #
    # df['V306-321_mean_with_zeros'] = df[[f'V{i}' for i in range(306, 322)]].mean(axis=1)
    # df['V306-321_std'] = df[[f'V{i}' for i in range(306, 322)]].mean(axis=1)
    #
    # df[f'V126-137_mean'] = \
    #     df[[f'V{i}' for i in range(126, 138)]].sum(axis=1) / \
    #     (df[[f'V{i}' for i in range(126, 138)]] > 0).sum(axis=1)
    #
    # df[f'V306-321_mean'] = \
    #     df[[f'V{i}' for i in range(306, 322)]].sum(axis=1) / \
    #     (df[[f'V{i}' for i in range(306, 322)]] > 0).sum(axis=1)
    #
    # df[[f'V{i}_diff' for i in range(279, 306)]] = df[[f'V{i}' for i in range(279, 306)]].diff()
    #
    for min_i, max_i in [
        (126, 137), #(306, 321)
    ]:
        for i in range(min_i, max_i + 1):
            df[f'V{i}_card1_mean'] = df\
                .groupby('subcard_categorical')[f'V{i}']\
                .transform('mean')

    df[[f'V{i}_diff1' for i in range(126, 138)]] = df\
        .groupby(['subcard_categorical', 'ProductCD', 'addr1'])[[f'V{i}' for i in range(126, 138)]]\
        .diff()

    return df


def add_shifted_features(df):

    columns_agg = [['card1', 'ProductCD']]
    column_to_shift = 'TransactionAmt'

    MAX_SHIFT = 5

    for columns in columns_agg:
        col_name = '_'.join(columns)

        grouping = df.groupby(columns)[column_to_shift]

        df[f'prev_{MAX_SHIFT}_{col_name}_{column_to_shift}'] = 0
        df[f'next_{MAX_SHIFT}_{col_name}_{column_to_shift}'] = 0

        for shift in range(1, MAX_SHIFT):

            df[f'prev_{MAX_SHIFT}_{col_name}_{column_to_shift}'] += grouping\
                .shift(shift) == df[column_to_shift]

            df[f'next_{MAX_SHIFT}_{col_name}_{column_to_shift}'] += grouping\
                .shift(-shift) == df[column_to_shift]

    return df


def relax_data(df_train, df_test, col):
    cv1 = pd.DataFrame(
        df_train[col]\
            .value_counts()
            .reset_index()
            .rename({col: 'train'}, axis=1)
    )
    cv2 = pd.DataFrame(
        df_test[col]\
            .value_counts()
            .reset_index()
            .rename({col: 'test'}, axis=1)
    )
    cv3 = pd.merge(cv1, cv2, on='index', how='outer')

    factor = len(df_test) / len(df_train)

    cv3['train'].fillna(0, inplace=True)
    cv3['test'].fillna(0, inplace=True)
    cv3['remove'] = False
    cv3['remove'] = cv3['remove'] | (cv3['train'] < len(df_train) / 10000)
    cv3['remove'] = cv3['remove'] | (factor * cv3['train'] < cv3['test'] / 3)
    cv3['remove'] = cv3['remove'] | (factor * cv3['train'] > 3 * cv3['test'])
    cv3['new'] = cv3.apply(
        lambda x: x['index']
        if x['remove'] == False
        else 0,
        axis=1
    )

    n_removed = sum(cv3['remove'])
    print(f'{col} Relaxed rows = {n_removed}')

    cv3['new'], _ = cv3['new'].factorize(sort=True)
    cv3.set_index('index', inplace=True)
    cc = cv3['new'].to_dict()
    df_train[col] = df_train[col].map(cc)
    df_test[col] = df_test[col].map(cc)

    return df_train, df_test


def norm_temporal_feature(df, column_to_norm, to_norm_by):

    __TMP = '__TMP'

    col_name_suffix = '_'.join(to_norm_by)

    df[__TMP] = df\
        .groupby(to_norm_by)[column_to_norm]\
        .transform('max')

    df[f'{column_to_norm}_normed_by_{col_name_suffix}'] = df[column_to_norm] \
                                                          / df[__TMP]

    df.drop(labels=[__TMP], axis=1, inplace=True)

    return df


def kfold_target_encoding(df, columns_to_group):

    folds = GroupKFold(n_splits=5)

    col_name = '_'.join(columns_to_group)
    col_name = f'target_encoded_{col_name}'

    df[col_name] = df[columns_to_group[0]]\
        .astype(str)\
        .copy()

    for i in columns_to_group[1:]:
        df[col_name] += '_'
        df[col_name] += df[i].astype(str)

    for train_ids, val_ids in folds.split(
            X=df,
            y=df['isFraud'],
            groups=df['subcard_categorical'].tolist()
    ):

        mapper = calc_smooth_mean(
            df.iloc[train_ids],
            columns_to_group,
            ['mean']
        )
        mapper = mapper['mean']

        df.iloc[val_ids, df.columns.get_loc(col_name)] = df\
            .iloc[val_ids][col_name]\
            .map(mapper).values

        df.iloc[val_ids, df.columns.get_loc(col_name)].fillna(
            df.iloc[train_ids]['isFraud'].mean(),
            inplace=True
        )

    df[col_name] = df[col_name].astype('float')

    return df


def get_base_target_encoding_mapper(df, columns_to_group):
    mapper = calc_smooth_mean(
        df,
        columns_to_group,
        ['mean']
    )
    mapper = mapper['mean']

    return mapper


def apply_target_encoder(df_train, df_test, columns_to_group):

    mapper = get_base_target_encoding_mapper(df_train, columns_to_group)

    col_name = '_'.join(columns_to_group)
    col_name = f'target_encoded_{col_name}'

    df_test[col_name] = df_test[columns_to_group[0]] \
        .astype(str) \
        .copy()

    for i in columns_to_group[1:]:
        df_test[col_name] += '_'
        df_test[col_name] += df_test[i].astype(str)

    df_test[col_name] = df_test[col_name].map(mapper).values
    df_test[col_name].fillna(
        df_train['isFraud'].mean(),
        inplace=True
    )

    df_test[col_name] = df_test[col_name].astype('float')

    return df_test


def values_normalization(df, col, clip=True, minmax=True):

    __TMP = '__TMP'

    df[__TMP] = df['TransactionDT_split'].astype(str) + '_' \
                + df['TransactionDT_dayOfMonth'].astype(str)

    new_col = col + '_' + 'GROUPED_BY_DAY'
    df_tmp = df[[col, __TMP]].copy()
    df_tmp[col] = df_tmp[col].astype(float)
    if clip:
        df_tmp[col] = df_tmp[col].clip(0)

    aggs = df_tmp.groupby(__TMP)[col].agg(['min', 'max', 'std', 'mean'])

    agg_max = aggs['max'].to_dict()
    agg_min = aggs['min'].to_dict()
    agg_std = aggs['std'].to_dict()
    agg_mean = aggs['mean'].to_dict()

    df['temp_min'] = df[__TMP].map(agg_max)
    df['temp_max'] = df[__TMP].map(agg_min)
    df['temp_std'] = df[__TMP].map(agg_std)
    df['temp_mean'] = df[__TMP].map(agg_mean)

    df[new_col + '_min_max'] = ((df[col] - df['temp_min']) / \
                                    (df['temp_max'] - df[
                                        'temp_min'])).astype(float)

    df[new_col + '_std_score'] = (df[col] - df['temp_mean']) / (df['temp_std'])

    df.drop(
        labels=['temp_min', 'temp_max', 'temp_std', 'temp_mean', __TMP],
        axis=1,
        inplace=True
    )

    return df


def advanced_D_processing(df):
    df['D8_not_same_day'] = np.where(df['D8'] >= 1, 1, 0)
    df['D8_D9_decimal_dist'] = df['D8'].fillna(0) \
                               - df['D8'].fillna(0).astype(int)
    df['D8_D9_decimal_dist'] = (
                                       (
                                               df['D8_D9_decimal_dist']
                                               - df['D9']
                                       ) ** 2
                               ) ** 0.5
    df['D8'] = df['D8'].fillna(-1).astype(int)

    return df


def advanced_M_processing(df):
    i_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

    df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)

    return df


def nan_count(df, columns):
    df['notnull_count'] = df[columns].notnull().sum(axis=1)
    return df


