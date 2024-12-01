import pandas as pd
import numpy as np

# Load the data (ensure to adjust the file path if needed)
file_path = 'c:/Users/jhkim/Desktop/bitcoin future prediction ai/btc_with_normalized_indicators.csv'
data = pd.read_csv(file_path)

def split_data(data, train_ratio=0.8, sequence_length=100):
    
    train_size = int(len(data) * train_ratio)
    train_set = data[:train_size]
    test_set = data[train_size:]
    
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data.iloc[i:i+seq_length].to_numpy()  # Ensure proper DataFrame slicing
            sequences.append(seq)
        return np.array(sequences)


   
    train_sequences = create_sequences(train_set, sequence_length)
    test_sequences = create_sequences(test_set, sequence_length)
    
    return train_sequences, test_sequences

train_set, test_set = split_data(data, train_ratio=0.8, sequence_length=100)

# print results to verify
# print(f"Train set size: {len(train_set)}")
# print(f"Test set size: {len(test_set)}")
# print(type(train_set))
# print(type(test_set))
print(train_set.shape)