import pandas as pd
import numpy as np

from feature_engineering import add_datetime_features, process_id_30, \
    process_id_33, emaildomain_features, count_features, smoothed_encodings, \
    encode_categorical_features, V_features_to_PCA
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
    train_test_joined = pd.concat([train, test], sort=True)

    train_test_joined = add_datetime_features(train_test_joined)
    train_test_joined = process_id_30(train_test_joined)
    train_test_joined = process_id_33(train_test_joined)
    train_test_joined = emaildomain_features(train_test_joined)
    train_test_joined = count_features(
        train_test_joined,
        columns_agg=[
            ['card1'],
            ['card2'],
            ['card3'],
            ['card4'],
            ['card5'],
            ['card6'],
            ['card1', 'card2', 'card3', 'card4'],
            ['addr1'],
            ['addr2'],
            ['TransactionDT_hour'],
            ['TransactionDT_dayOfWeek'],
            ['TransactionDT_dayOfMonth'],
            ['TransactionDT_weekOfMonth'],
            ['DeviceType'],
            ['DeviceInfo'],
            # derived
            ['OS_NAME'],
            ['id_30'],
            ['ProductCD']
        ]
    )

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
            ['card1', 'card2', 'card3', 'card4'],
            ['addr1'],
            ['addr2'],
            ['TransactionDT_hour'],
            ['TransactionDT_dayOfWeek'],
            ['TransactionDT_dayOfMonth'],
            ['TransactionDT_weekOfMonth'],
            ['DeviceType'],
            ['DeviceInfo'],
            # derived
            ['OS_NAME'],
            ['id_30'],
            ['ProductCD']
        ],
        TARGET,
        funcs=['mean'],
        train_size=sum(train_test_joined['TransactionDT_split'].isin(bounds))
    )

    # mean, std encodings
    for val in ['TransactionAmt', 'id_02']:

        train_test_joined = smoothed_encodings(
            train_test_joined,
            [
                ['card1'],
                ['card2'],
                ['card3'],
                ['card4'],
                ['card5'],
                ['card6'],
                ['card1', 'card2', 'card3', 'card4'],
                ['addr1'],
                ['addr2'],
                ['TransactionDT_hour'],
                ['TransactionDT_dayOfWeek'],
                ['TransactionDT_dayOfMonth'],
                ['TransactionDT_weekOfMonth'],
                ['DeviceType'],
                ['DeviceInfo'],
                # derived
                ['OS_NAME'],
                ['id_30'],
                ['ProductCD']
            ],
            val
        )

    #train_test_joined = V_features_to_PCA(train_test_joined)

    train_test_joined, encoders = encode_categorical_features(
        train_test_joined,
        CATEGORICAL_FEATURES
    )

    return train_test_joined
