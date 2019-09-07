import datetime


START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

TARGET = 'isFraud'

CATEGORICAL_FEATURES = [
    'ProductCD',
    'card1',
    'card2',
    'card3',
    'card4',
    'card5',
    'card6',
    'addr1',
    'addr2',
    'P_emaildomain',
    'R_emaildomain',
    'M1',
    'M2',
    'M3',
    'M4',
    'M5',
    'M6',
    'M7',
    'M8',
    'M9',
    # identity
    'DeviceType',
    'DeviceInfo',
    # derived
    'OS_NAME',
    'OS_V0',
    'OS_V1',
    'OS_V2',
    #
    'P_emaildomain_1',
    'P_emaildomain_2',
    'P_emaildomain_3',
    'R_emaildomain_1',
    'R_emaildomain_2',
    'R_emaildomain_3',
    #
    'id_12',
    'id_15',
    'id_16',
    'id_23',
    'id_27',
    'id_28',
    'id_29',
    'id_30',
    'id_31',
    'id_33',
    'id_34',
    'id_35',
    'id_36',
    'id_37',
    'id_38'
]# + [f'id_{i}' for i in range(12, 39)]


COLUMNS_TO_REMOVE = [
    'TransactionID',
    TARGET,
    'TransactionDT_split',
]