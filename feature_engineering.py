import numpy as np
import pandas as pd
import datetime

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from settings import START_DATE


def calc_smooth_mean(
        df,
        columns_agg,
        funcs=['mean', 'std', np.nanmedian],
        on='isFraud',
        m=10,
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
        train_size=None
):
    if train_size is None:
        train_size = df.shape[0]

    for col_agg in columns_agg:
        encoders = calc_smooth_mean(
            df.iloc[:int(train_size)],
            col_agg,
            funcs=funcs,
            on=column_value
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
        columns_agg
):
    for c_agg in columns_agg:

        col_name = '_'.join(c_agg)

        df[f'{col_name}_count'] = df[c_agg]\
            .groupby(c_agg)[c_agg[0]]\
            .transform('count')

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


def add_datetime_features(df):

    tr_dt = 'TransactionDT'

    df[f'{tr_dt}_to_datetime'] = df[tr_dt].apply(
        lambda x: START_DATE + datetime.timedelta(seconds=x)
    )

    df[f'{tr_dt}_year'] = df[f'{tr_dt}_to_datetime'].dt.year - 2017
    df[f'{tr_dt}_month'] = df[f'{tr_dt}_to_datetime'].dt.month
    df[f'{tr_dt}_dayOfMonth'] = df[f'{tr_dt}_to_datetime'].dt.day
    df[f'{tr_dt}_dayOfWeek'] = df[f'{tr_dt}_to_datetime'].apply(
        lambda x: x.weekday())
    df[f'{tr_dt}_weekOfMonth'] = (df[f'{tr_dt}_to_datetime'].dt.day - 1) // 7 + 1
    df[f'{tr_dt}_hour'] = df[f'{tr_dt}_to_datetime'].dt.hour
    #df[f'{tr_dt}_minute'] = df[f'{tr_dt}_to_datetime'].dt.minute
    #df[f'{tr_dt}_second'] = df[f'{tr_dt}_to_datetime'].dt.second
    df[f'{tr_dt}_split'] = (df[f'{tr_dt}_to_datetime'].dt.year - 2017) * 12 + \
                                df[f'{tr_dt}_to_datetime'].dt.month

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

        if df[f].dtype in ('float64', 'object'):
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

    return df


def V_features_to_PCA(df):

    pca_reducer = PCA(n_components=0.85)

    V_features = [f'V{i}' for i in range(1, 340)]

    transformed_values = pca_reducer.fit_transform(
        StandardScaler().fit_transform(
            df[V_features].fillna(0.0)
        )
    )

    size = transformed_values.shape[1]

    pca_df = pd.DataFrame(
        data=transformed_values,
        columns=[f'PCA_V{i}' for i in range(size)],
        index=df.index
    )

    df.drop(labels=V_features, axis=1, inplace=True)

    df = pd.concat([df, pca_df], axis=1)

    return df
