# main.py
import os
from fetch_data import fetch_futures_data_with_indicators
from normalize_ohlcv import normalize_ohlcv

def main():
    # Fetch data with indicators
    df = fetch_futures_data_with_indicators(symbol='BTC/USDT', timeframe='15m', total_candles=182000)

    # Normalize the OHLCV data
    df_normalized = normalize_ohlcv(df)

    # Save the normalized data to a CSV file
    df_normalized.to_csv('btc_with_normalized_indicators.csv', index=False)

    # Optionally, save the data as a pickle file
    df_normalized.to_pickle('btc_with_normalized_indicators.pkl')

    # Display the first few rows of the normalized DataFrame
    print(df_normalized.head())

if __name__ == '__main__':
    main()
