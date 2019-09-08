import pandas as pd
import numpy as np
import datetime

from feature_engineering import add_datetime_features, process_id_30, \
    process_id_33, emaildomain_features, count_features, smoothed_encodings, \
    encode_categorical_features, V_features_to_PCA, D_features_to_PCA, \
    C_features_to_PCA, exchange_rate_took_place_feature, generate_uid_features
from settings import CATEGORICAL_FEATURES, TARGET


def generate_features_time_series(train, test, bounds=(12, 13, 14, 15, 16)):

    '''

    :param train:
    :param test:
    :param mode: train or inference
    :return:
    '''

    train_size = train.shape[0]
    test_size = test.shape[0]

    print('Starting', datetime.datetime.now())

    train_test_joined = pd.concat([train, test], sort=True)
    print('Concatted', datetime.datetime.now())

    train_test_joined = add_datetime_features(train_test_joined)
    train_test_joined = exchange_rate_took_place_feature(train_test_joined)
    print('DT FEATURES', datetime.datetime.now())
    train_test_joined = process_id_30(train_test_joined)
    train_test_joined = process_id_33(train_test_joined)
    train_test_joined = emaildomain_features(train_test_joined)
    #train_test_joined = generate_uid_features(train_test_joined)
    print('ids, emaildomain', datetime.datetime.now())

    train_test_joined = count_features(
        train_test_joined,
        columns_agg=[
            ['card1'],
            ['card2'],
            ['card3'],
            ['card4'],
            ['card5'],
            ['card6'],
            ['addr1'],
            ['addr2'],
            ['TransactionDT_hour'],
            ['card1', 'TransactionDT_hour'],
            ['TransactionDT_dayOfMonth'],
            ['TransactionDT_weekOfMonth'],
            ['DeviceInfo'],
            # derived
            ['OS_NAME'],
            ['P_emaildomain'],
            ['R_emaildomain'],
            ['card1', 'TransactionAmt'],
            ['card1', 'P_emaildomain'],
            ['card1', 'R_emaildomain'],
            ['card1', 'addr1'],
            ['card1', 'addr2'],
            ['card1', 'TransactionDT_dayOfWeek'],
            ['card1', 'DeviceInfo'],
            ['card1', 'dist1'],
            ['card1', 'dist2'],
            ['card1', 'OS_NAME'],
            ['card1', 'C13'],
            ['card1', 'C14'],
            ['card1', 'C1']
        ] + [[f'id_{i}'] for i in range(12, 39)],
        with_typical_for_user=True
    )

    train_test_joined = count_features(
        train_test_joined,
        columns_agg=[
            ['card1', 'TransactionDT_split', 'TransactionDT_dayOfMonth'],
            ['card1', 'TransactionDT_split', 'TransactionDT_dayOfMonth', 'TransactionDT_hour'],
        ],
        with_typical_for_user=False
    )

    print('Count features', datetime.datetime.now())

    '''
    # target encoding
    train_test_joined = smoothed_encodings(
        train_test_joined,
        [
            ['TransactionDT_hour'],
            ['TransactionDT_dayOfWeek'],
            ['TransactionDT_dayOfMonth'],
        ],
        TARGET,
        funcs=['mean'],
        train_size=sum(train_test_joined['TransactionDT_split'].isin(bounds)),
        m=100
    )
    '''

    print('target encoding', datetime.datetime.now())

    # mean, std encodings
    for vals, columns_agg, funcs, m in [
        (
                ['TransactionAmt'],
                [
                    ['card1'],
                    ['TransactionDT_hour'],
                    ['TransactionDT_dayOfMonth'],
                    ['TransactionDT_weekOfMonth'],
                    # derived
                    ['card1', 'TransactionDT_hour'],
                    ['card1', 'TransactionDT_dayOfWeek']
                ],
                ['mean', 'std', np.nanmedian],
                3
        ),
        (
                ['dist1', 'dist2'],
                [
                    ['card1'],
                ],
                ['mean', 'std', np.nanmedian],
                0
        ),
        (
                ['TransactionDT_hour'],
                [
                    ['card1'],
                ],
                ['nunique', 'mean'],
                0
        ),
        (
                ['OS_NAME', 'C13', 'C14', 'C1'],
                [
                    ['card1'],
                ],
                ['nunique'],
                0
        ),
        (
            ['TransactionAmt'],
            [
                ['card1', 'TransactionDT_split', 'TransactionDT_dayOfMonth'],
                ['card1', 'TransactionDT_split', 'TransactionDT_dayOfMonth', 'TransactionDT_hour']
            ],
            ['sum', 'mean'],
            0
        )
    ]:

        for val in vals:
            train_test_joined = smoothed_encodings(
                train_test_joined,
                columns_agg,
                val,
                funcs=funcs,
                m=m
            )
    print('Mean Encoding', datetime.datetime.now())

    train_test_joined = V_features_to_PCA(train_test_joined)
    train_test_joined = D_features_to_PCA(train_test_joined)
    train_test_joined = C_features_to_PCA(train_test_joined)

    train_test_joined, encoders = encode_categorical_features(
        train_test_joined,
        CATEGORICAL_FEATURES
    )
    print('Encoders', datetime.datetime.now())

    return train_test_joined


def generate_features_time_series_inference(
        train, test, bounds=(12, 13, 14, 15, 16, 17)
):

    '''

    :param train:
    :param test:
    :param mode: train or inference
    :return:
    '''

    train_size = train.shape[0]
    test_size = test.shape[0]

    print('Starting', datetime.datetime.now())

    train_test_joined = pd.concat([train, test], sort=True)
    print('Concatted', datetime.datetime.now())

    train_test_joined = add_datetime_features(train_test_joined)
    train_test_joined = exchange_rate_took_place_feature(train_test_joined)
    print('DT FEATURES', datetime.datetime.now())
    train_test_joined = process_id_30(train_test_joined)
    train_test_joined = process_id_33(train_test_joined)
    train_test_joined = emaildomain_features(train_test_joined)
    #train_test_joined = generate_uid_features(train_test_joined)
    print('ids, emaildomain', datetime.datetime.now())

    train_test_joined = count_features(
        train_test_joined,
        columns_agg=[
            ['card1'],
            ['card2'],
            ['card6'],
            ['P_emaildomain']
        ]
    )
    print('Count features', datetime.datetime.now())

    '''
    # target encoding
    train_test_joined = smoothed_encodings(
        train_test_joined,
        [
            ['card1'],
            ['card2'],
            ['card3'],
            ['card4'],
            ['card5'],
            ['card6'],
            #['card1', 'card2', 'card3', 'card4'],
            ['addr1'],
            ['addr2'],
            ['TransactionDT_hour'],
            ['TransactionDT_dayOfWeek'],
            ['TransactionDT_dayOfMonth'],
            ['TransactionDT_weekOfMonth'],
            #['TransactionDT_hour', 'TransactionDT_dayOfWeek'],
            ['DeviceType'],
            ['DeviceInfo'],
            # derived
            ['OS_NAME'],
            #['id_30'],
            #['id_12'],
            #['id_15'],
            #['id_16'],
            #['id_23'],
            #['id_27'],
            #['id_28'],
            #['id_29'],
            #['id_30'],
            #['id_31'],
            #['id_33'],
            #['id_34'],
            #['id_35'],
            #['id_36'],
            #['id_37'],
            #['id_38'],
            ['P_emaildomain'],
            ['R_emaildomain'],
            #'P_emaildomain_1'],
            #['R_emaildomain_1'],
            #['M1'],
            #['M2'],
            #['M3'],
            #['M4'],
            #['M5'],
            #['M6'],
            #['M7'],
            #['M8'],
            #['M9'],
            ['ProductCD'],
            ['bank_type'],
            ['uid'],
            ['uid2'],
            ['uid3'],
            ['uid4'],
            ['uid5']
        ],
        TARGET,
        funcs=['mean'],
        train_size=sum(train_test_joined['TransactionDT_split'].isin(bounds))
    )
    '''
    print('target encoding', datetime.datetime.now())

    # mean, std encodings
    for vals, columns_agg, funcs, m in [
        (
                ['TransactionAmt'],
                [
                    ['card1'],
                    ['card2'],
                    ['TransactionDT_hour'],
                ],
                ['mean', 'std', np.nanmedian],
                5
        ),
        (
                ['dist1', 'dist2'],
                [
                    ['card1'],
                ],
                ['mean', 'std'],
                0
        ),
        (
                ['TransactionDT_hour'],
                [
                    ['card1'],
                ],
                ['nunique', 'mean'],
                0
        )
    ]:

        for val in vals:
            train_test_joined = smoothed_encodings(
                train_test_joined,
                columns_agg,
                val,
                funcs=funcs,
                m=m
            )
    print('Mean Encoding', datetime.datetime.now())

    #train_test_joined = V_features_to_PCA(train_test_joined)
    #train_test_joined = D_features_to_PCA(train_test_joined)
    #train_test_joined = C_features_to_PCA(train_test_joined)

    train_test_joined, encoders = encode_categorical_features(
        train_test_joined,
        CATEGORICAL_FEATURES
    )
    print('Encoders', datetime.datetime.now())

    return train_test_joined
