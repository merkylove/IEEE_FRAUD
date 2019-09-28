import pandas as pd
import numpy as np
import datetime

from feature_engineering import add_datetime_features, process_id_30, \
    process_id_33, emaildomain_features, count_features, smoothed_encodings, \
    encode_categorical_features, V_features_to_PCA, D_features_to_PCA, \
    C_features_to_PCA, exchange_rate_took_place_feature, add_is_null_features, \
    device_to_group, remove_rare_values, base_transaction_delta_features, \
    V_groups_to_nan, advanced_V_processing, add_shifted_features, relax_data, \
    extract_registration_date
from settings import CATEGORICAL_FEATURES, TARGET


def generate_features_time_series(train, test, bounds=(12, 13, 14, 15, 16, 17)):

    '''

    :param train:
    :param test:
    :param mode: train or inference
    :return:
    '''

    train_size = train.shape[0]
    test_size = test.shape[0]

    print('NEW')

    print('Starting', datetime.datetime.now())

    # for i in [
    #     'ProductCD',
    #     'card1',
    #     'card2',
    #     'card3',
    #     'card4',
    #     'card5',
    #     'card6',
    #     'addr1',
    #     'addr2',
    # ]:
    #     train, test = relax_data(train, test, i)

    train = extract_registration_date(train)
    test = extract_registration_date(test)
    train = base_transaction_delta_features(train)
    test = base_transaction_delta_features(test)
    train = advanced_V_processing(train)
    test = advanced_V_processing(test)

    train_test_joined = pd.concat([train, test], sort=True)
    print('Concatted', datetime.datetime.now())

    train_test_joined = add_datetime_features(train_test_joined)

    #train_test_joined = train_test_joined[train_test_joined['TransactionDT_split'] > 12]

    train_test_joined = exchange_rate_took_place_feature(train_test_joined)
    print('DT FEATURES', datetime.datetime.now())
    train_test_joined = process_id_30(train_test_joined)
    #train_test_joined = process_id_33(train_test_joined)
    train_test_joined = emaildomain_features(train_test_joined)
    train_test_joined = add_is_null_features(train_test_joined)
    train_test_joined = device_to_group(train_test_joined)
    train_test_joined = V_groups_to_nan(train_test_joined)
    train_test_joined = add_shifted_features(train_test_joined)
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
            ['device_name'],
            # derived
            ['P_emaildomain'],
            ['R_emaildomain'],
            ['card1', 'addr1'],
            ['card2', 'addr1'],
            ['card3', 'addr1'],
            ['card4', 'addr1'],
            ['card6', 'addr1'],
            ['card1', 'P_emaildomain'],
            ['card1', 'R_emaildomain'],
            ['card1', 'TransactionDT_hour'],
            ['card1', 'is_foreign'],
            ['card1', 'ProductCD'],
            ['card1', 'ProductCD', 'addr1'],
            ['ProductCD', 'addr1'],
            ['ProductCD'],
            ['card1', 'subcard_categorical'],
            ['card1', 'subcard_categorical', 'subcard_categorical_D4'],
            ['card1', 'subcard_categorical', 'DeviceInfo'],
            ['card1', 'subcard_categorical', 'P_emaildomain'],
            ['card1', 'subcard_categorical', 'P_emaildomain', 'subcard_categorical_D4'],
            ['card1', 'subcard_categorical', 'ProductCD'],
            ['card1', 'subcard_categorical', 'ProductCD', 'subcard_categorical_D4'],
            ['card1', 'subcard_categorical', 'addr1'],
            ['card1', 'subcard_categorical', 'id_20'],
            ['card1', 'subcard_categorical', 'id_19'],
            ['card1', 'subcard_categorical', 'id_31'],
            ['card1', 'subcard_categorical', 'card4']
        ] + [['card1', f'C{i}'] for i in range(1, 15)],
        with_typical_for_user=True
    )

    train_test_joined = count_features(
        train_test_joined,
        columns_agg=[
            [
                'card1',
                'subcard_categorical',
                'TransactionDT_split',
                'TransactionDT_dayOfMonth'
            ],
            [
                'card1',
                'subcard_categorical',
                'TransactionDT_split',
                'TransactionDT_dayOfMonth',
                'TransactionDT_hour'
            ],
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
                    ['card2'],
                    ['card3'],
                    ['card5'],
                    ['TransactionDT_hour'],
                    ['TransactionDT_dayOfMonth'],
                    ['TransactionDT_weekOfMonth'],
                    # derived
            ['card1', 'subcard_categorical'],
            ['card1', 'subcard_categorical', 'subcard_categorical_D4'],
            ['card1', 'subcard_categorical', 'DeviceInfo'],
            ['card1', 'subcard_categorical', 'P_emaildomain'],
            ['card1', 'subcard_categorical', 'P_emaildomain', 'subcard_categorical_D4'],
            ['card1', 'subcard_categorical', 'ProductCD'],
            ['card1', 'subcard_categorical', 'ProductCD', 'subcard_categorical_D4'],
            ['card1', 'subcard_categorical', 'addr1'],
            ['card1', 'subcard_categorical', 'id_20'],
            ['card1', 'subcard_categorical', 'id_19'],
            ['card1', 'subcard_categorical', 'id_31'],
            ['card1', 'subcard_categorical', 'card4']
                ],
                ['mean', 'std', np.nanmedian],
                10
        ),
        (
               ['dist1'],
               [
                   ['card1'],
                   ['card1', 'subcard_categorical'],
            ['card1', 'subcard_categorical', 'DeviceInfo']
               ],
               ['mean', 'std', np.nanmedian],
               0
        ),
        # (
        #         ['C13', 'C14', 'C1'],
        #         [
        #             ['card1'],
        #         ],
        #         ['mean', 'std', np.nanmedian],
        #         0
        # ),
        (
                ['TransactionDT_hour'],
                [
                    ['card1'],
                ],
                ['mean'],
                0
        ),
        (
                ['card2', 'P_emaildomain', 'R_emaildomain', 'ProductCD'],
                [
                    ['card1'],
                ],
                ['nunique'],
                0
        ),
        (
                ['P_emaildomain', 'R_emaildomain', 'ProductCD'],
                [
                    ['card2'],
                ],
                ['nunique'],
                0
        ),
        (
                ['P_emaildomain', 'R_emaildomain', 'ProductCD'],
                [
                    ['addr1'],
                ],
                ['nunique'],
                0
        ),
        (
                ['P_emaildomain'],
                [
                    ['R_emaildomain'],
                ],
                ['nunique'],
                0
        ),
        (
                ['R_emaildomain'],
                [
                    ['P_emaildomain'],
                ],
                ['nunique'],
                0
        ),
        (
                ['card1'],
                [
                    ['card2'],
                ],
                ['nunique'],
                0
        ),
        (
                ['card1'],
                [
                    ['card4'],
                ],
                ['nunique'],
                0
        ),
        (
                ['card1'],
                [
                    ['card6'],
                ],
                ['nunique'],
                0
        ),
        (
                ['card1'],
                [
                    ['card2', 'card3'],
                ],
                ['nunique'],
                0
        ),
        (
                ['card1'],
                [
                    ['addr1'],
                ],
                ['nunique'],
                0
        ),
        (
            ['TransactionAmt'],
            [
                #['card1', 'TransactionDT_split', 'TransactionDT_dayOfMonth'],
                ['card1', 'TransactionDT_split', 'TransactionDT_dayOfMonth', 'TransactionDT_hour'],
            ],
            ['sum'],
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

    # train_test_joined = remove_rare_values(
    #     train_test_joined,
    #     train.shape[0],
    #     ['card1']
    # )

    train_test_joined, encoders = encode_categorical_features(
        train_test_joined,
        CATEGORICAL_FEATURES
    )
    print('Encoders', datetime.datetime.now())

    return train_test_joined, encoders
