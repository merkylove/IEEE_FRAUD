import pandas as pd


def read_data():
    train_transaction = pd.read_csv(
        'train_transaction.csv',
        index_col='TransactionID'
    )
    train_identity = pd.read_csv(
        'train_identity.csv',
        index_col='TransactionID'
    )
    test_transaction = pd.read_csv(
        'test_transaction.csv',
        index_col='TransactionID'
    )
    test_identity = pd.read_csv(
        'test_identity.csv',
        index_col='TransactionID'
    )
    sample_submission = pd.read_csv('sample_submission.csv')

    train = pd.merge(
        train_transaction,
        train_identity,
        on='TransactionID',
        how='left'
    )
    test = pd.merge(
        test_transaction,
        test_identity,
        on='TransactionID',
        how='left'
    )

    return train, test, sample_submission