# normalize_ohlcv.py
from sklearn.preprocessing import MinMaxScaler

def normalize_ohlcv(df):
    # Columns to normalize (OHLCV)
    columns_to_normalize = ['open', 'high', 'low', 'close', 'volume']

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Apply scaling to the selected columns
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    return df
