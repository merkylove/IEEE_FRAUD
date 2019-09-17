import pandas as pd
import numpy as np
import datetime

from feature_engineering import add_datetime_features, process_id_30, \
    process_id_33, emaildomain_features, count_features, smoothed_encodings, \
    encode_categorical_features, V_features_to_PCA, D_features_to_PCA, \
    C_features_to_PCA, exchange_rate_took_place_feature, add_is_null_features, \
    device_to_group, remove_rare_values, base_transaction_delta_features, \
    V_groups_to_nan, advanced_V_processing, add_shifted_features, relax_data, \
    extract_registration_date, norm_temporal_feature, kfold_target_encoding, \
    process_id_31, values_normalization, advanced_D_processing, \
    advanced_M_processing, nan_count
from settings import CATEGORICAL_FEATURES, TARGET, \
    USELESS_FEATURES_ACCORDING_TO_PERMUTATIONS


def generate_features_time_series(train, test, bounds=(12, 13, 14, 15, 16, 17)):

    '''

    :param train:
    :param test:
    :param mode: train or inference
    :return:
    '''

    train_size = train.shape[0]
    test_size = test.shape[0]

    BASE_FIELDS = test.columns.tolist()
    BASE_FIELDS = set(BASE_FIELDS) - set(['TransactionDT', 'TransactionID'])
    BASE_FIELDS = list(BASE_FIELDS)

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

    for i in [
            'D1',
            'D2',
            'D3',
            'D4',
            'D5',
            'D6',
            'D7',
            'D8',
            'D10',
            'D11',
            'D12',
            'D13',
            'D14',
            'D15',
            'C1',
            'C2',
            'C4',
            'C5',
            'C6',
            'C7',
            'C8',
            'C9',
            'C10',
            'C11',
            'C12',
            'C13',
            'C14',
            'subcard_reg_timestamp'
        ]:

        train_test_joined = values_normalization(train_test_joined, i)

    train_test_joined = exchange_rate_took_place_feature(train_test_joined)
    print('DT FEATURES', datetime.datetime.now())
    train_test_joined = process_id_31(train_test_joined)
    train_test_joined = process_id_30(train_test_joined)
    train_test_joined = process_id_33(train_test_joined)
    train_test_joined = emaildomain_features(train_test_joined)
    train_test_joined = device_to_group(train_test_joined)
    #train_test_joined = V_groups_to_nan(train_test_joined)
    train_test_joined = add_shifted_features(train_test_joined)
    #train_test_joined = generate_uid_features(train_test_joined)
    print('ids, emaildomain', datetime.datetime.now())

    train_test_joined = count_features(
        train_test_joined,
        columns_agg=[
            ['card3'],
            ['TransactionDT_hour'],
            ['R_emaildomain'],
            ['P_emaildomain'],
            ['card2', 'addr1'],
            ['ProductCD'],
            ['ProductCD', 'addr1'],
            ['subcard_categorical', 'R_emaildomain'],
            ['subcard_categorical', 'P_emaildomain'],
            ['subcard_categorical'],
            ['card2'],
            ['card4'],
            ['card5'],
            ['card6'],
            ['addr1'],
            ['addr2'],
            ['TransactionDT_dayOfMonth'],
            ['TransactionDT_weekOfMonth'],
            ['DeviceInfo'],
            ['device_name'],
            ['card2', 'addr1'],
            ['card3', 'addr1'],
            ['card4', 'addr1'],
            ['card6', 'addr1'],
            ['id_19'],
            ['id_20'],
            ['id_31'],
        ]
                    + [[f'C{i}' for i in range(1, 15)]],
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
            ['card1', 'subcard_categorical', 'ProductCD'],
            ['card1', 'subcard_categorical', 'ProductCD', 'addr1'],
            ['card1', 'subcard_categorical', 'addr1'],
            ['card1', 'addr1'],
            ['card1', 'TransactionDT_hour'],
            ['subcard_categorical', 'card2'],
            ['subcard_categorical', 'card3'],
            ['subcard_categorical', 'card4'],
            ['subcard_categorical', 'card6'],
            ['card1'],
            ['subcard_categorical', 'device_name'],
            ['subcard_categorical', 'DeviceInfo'],
            [
                'card1',
                'subcard_categorical',
                'TransactionDT_split',
                'TransactionDT_dayOfMonth',
                'TransactionDT_hour'
            ],
            ['card1', 'P_emaildomain'],
            ['card1', 'R_emaildomain'],
            ['card1', 'TransactionDT_hour'],
            ['card1', 'ProductCD'],
            ['card1', 'ProductCD', 'addr1'],
        ]
                    + [['subcard_categorical', f'C{i}'] for i in range(1, 15)]
                    + [['card1', f'C{i}'] for i in range(1, 15)],
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
                ],
                ['mean', 'std', np.nanmedian],
                10
        ),
        (
               ['dist1'],
               [
                   ['card1'],
               ],
               ['std', np.nanmedian],
               0
        ),
        (
                ['dist1'],
                [
                    ['card1', 'subcard_categorical']
                ],
                ['mean', 'std', np.nanmedian],
                0
        ),
        (
                ['TransactionDT_hour'],
                [
                    ['card1'],
                    ['subcard_categorical']
                ],
                ['mean'],
                0
        ),
        (
                ['P_emaildomain', 'R_emaildomain'],
                [
                    ['subcard_categorical']
                ],
                ['nunique'],
                0
        ),
        (
                ['R_emaildomain'],
                [
                    ['card1'],
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

    train_test_joined = advanced_D_processing(train_test_joined)
    train_test_joined = advanced_M_processing(train_test_joined)
    train_test_joined = nan_count(train_test_joined, BASE_FIELDS)

    train_test_joined, encoders = encode_categorical_features(
        train_test_joined,
        [
            i for i in CATEGORICAL_FEATURES
            if i in train_test_joined.columns
            #if i not in USELESS_FEATURES_ACCORDING_TO_PERMUTATIONS
        ]
    )
    print('Encoders', datetime.datetime.now())

    return train_test_joined, encoders
