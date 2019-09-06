import pandas as pd
import numpy as np

from feature_engineering import add_datetime_features, process_id_30, \
    process_id_33, emaildomain_features, count_features, smoothed_encodings, \
    encode_categorical_features
from settings import CATEGORICAL_FEATURES, TARGET


def generate_features(train, test, mode='train'):

    '''

    :param train:
    :param test:
    :param mode: train or inference
    :return:
    '''

    bound = 16 if mode == 'train' else 17

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
            ['TransactionDT_dayOfWeek', 'addr1'],
            ['TransactionDT_weekOfMonth', 'addr1'],
            ['M1'],
            ['M2'],
            ['M3'],
            ['M4'],
            ['M5'],
            ['M6'],
            ['M7'],
            ['M8'],
            ['M9']
        ]
    )

    # target encoding
    train_test_joined = smoothed_encodings(
        train_test_joined,
        [
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
        ],
        TARGET,
        funcs=['mean', 'std'],
        train_size=sum(train_test_joined['TransactionDT_split'] <= bound)
    )

    # mean, std encodings
    for val in ['TransactionAmt', 'id_02', 'D15', 'C13']:

        train_test_joined = smoothed_encodings(
            train_test_joined,
            [
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
            ],
            val
        )

    train_test_joined, encoders = encode_categorical_features(
        train_test_joined,
        CATEGORICAL_FEATURES
    )

    return train_test_joined
