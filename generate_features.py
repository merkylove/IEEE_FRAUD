import pandas as pd
import datetime

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

    print('Starting', datetime.datetime.now())

    train_test_joined = pd.concat([train, test], sort=True)
    print('Concatted', datetime.datetime.now())

    train_test_joined = add_datetime_features(train_test_joined)
    print('DT FEATURES', datetime.datetime.now())
    train_test_joined = process_id_30(train_test_joined)
    train_test_joined = process_id_33(train_test_joined)
    train_test_joined = emaildomain_features(train_test_joined)
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
            ['card1', 'card2', 'card3', 'card4'],
            ['addr1'],
            ['addr2'],
            ['TransactionDT_hour'],
            ['TransactionDT_dayOfWeek'],
            ['TransactionDT_dayOfMonth'],
            ['TransactionDT_weekOfMonth'],
            ['TransactionDT_hour', 'TransactionDT_dayOfWeek'],
            ['DeviceType'],
            ['DeviceInfo'],
            # derived
            ['OS_NAME'],
            ['id_30'],
            ['id_12'],
            ['id_15'],
            ['id_16'],
            ['id_23'],
            ['id_27'],
            ['id_28'],
            ['id_29'],
            ['id_30'],
            ['id_31'],
            ['id_33'],
            ['id_34'],
            ['id_35'],
            ['id_36'],
            ['id_37'],
            ['id_38'],
            ['P_emaildomain'],
            ['R_emaildomain'],
            ['M1'],
            ['M2'],
            ['M3'],
            ['M4'],
            ['M5'],
            ['M6'],
            ['M7'],
            ['M8'],
            ['M9'],
            ['ProductCD'],
        ]
    )
    print('Count features', datetime.datetime.now())

    # target encoding
    train_test_joined = smoothed_encodings(
        train_test_joined,
        [
            #['card1'],
            #['card2'],
            #['card3'],
            #['card4'],
            #['card5'],
            #['card6'],
            #['card1', 'card2', 'card3', 'card4'],
            ['TransactionDT_hour'],
            ['TransactionDT_dayOfWeek'],
            ['TransactionDT_dayOfMonth'],
            ['TransactionDT_weekOfMonth'],
            ['TransactionDT_hour', 'TransactionDT_dayOfWeek'],
            ['DeviceType'],
            ['DeviceInfo'],
            # derived
            ['OS_NAME'],
            ['id_30'],
            ['id_12'],
            ['id_15'],
            ['id_16'],
            ['id_23'],
            ['id_27'],
            ['id_28'],
            ['id_29'],
            ['id_30'],
            ['id_31'],
            ['id_33'],
            ['id_34'],
            ['id_35'],
            ['id_36'],
            ['id_37'],
            ['id_38'],
            ['P_emaildomain'],
            ['R_emaildomain'],
            ['M1'],
            ['M2'],
            ['M3'],
            ['M4'],
            ['M5'],
            ['M6'],
            ['M7'],
            ['M8'],
            ['M9'],
            ['ProductCD'],
        ],
        TARGET,
        funcs=['mean'],
        train_size=sum(train_test_joined['TransactionDT_split'].isin(bounds))
    )
    print('target encoding', datetime.datetime.now())

    # mean, std encodings
    for val in ['TransactionAmt']:

        train_test_joined = smoothed_encodings(
            train_test_joined,
            [
                #['card1'],
                #['card2'],
                #['card3'],
                #['card4'],
                #['card5'],
                #['card6'],
                #['card1', 'card2', 'card3', 'card4'],
                ['addr1'],
                ['addr2'],
                ['TransactionDT_hour'],
                ['TransactionDT_dayOfWeek'],
                ['TransactionDT_dayOfMonth'],
                ['TransactionDT_weekOfMonth'],
                ['TransactionDT_hour', 'TransactionDT_dayOfWeek'],
                ['DeviceType'],
                ['DeviceInfo'],
                # derived
                ['OS_NAME'],
                ['id_30'],
                ['id_12'],
                ['id_15'],
                ['id_16'],
                ['id_23'],
                ['id_27'],
                ['id_28'],
                ['id_29'],
                ['id_30'],
                ['id_31'],
                ['id_33'],
                ['id_34'],
                ['id_35'],
                ['id_36'],
                ['id_37'],
                ['id_38'],
                ['P_emaildomain'],
                ['R_emaildomain'],
                ['M1'],
                ['M2'],
                ['M3'],
                ['M4'],
                ['M5'],
                ['M6'],
                ['M7'],
                ['M8'],
                ['M9'],
                ['ProductCD'],
            ],
            val
        )
    print('Mean Encoding', datetime.datetime.now())

    #train_test_joined = V_features_to_PCA(train_test_joined)

    train_test_joined, encoders = encode_categorical_features(
        train_test_joined,
        CATEGORICAL_FEATURES
    )
    print('Encoders', datetime.datetime.now())

    return train_test_joined
