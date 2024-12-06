# import pandas as pd
# from splitdata import split_data
# from normalize_ohlcv import normalize_ohlcv
# from deeplearningmodel import build_model
# import numpy as np
# from tensorflow.keras.callbacks import EarlyStopping

# # Step 1: Load Data
# file_path = 'https://raw.githubusercontent.com/koreakimbap/bitcoin_prediction_ai/main/btc_with_normalized_indicators.csv'
# data = pd.read_csv(file_path)

# # Step 2: Split Data (Train/Test)
# train_set, test_set = split_data(data, train_ratio=0.8, sequence_length=100)

# # Step 3: Prepare Data for LSTM (Create X and y)
# # def prepare_lstm_data(data, sequence_length, predict_length):
# #     X, y = [], []
# #     for i in range(len(data) - sequence_length - predict_length + 1):
# #         X.append(data.iloc[i:i + sequence_length].values)  # Input sequences
# #         y.append(data.iloc[i + sequence_length:i + sequence_length + predict_length]['close'].values)  # Target sequences
# #     return X, y
# def prepare_lstm_data(data, sequence_length, predict_length):
#     X, y = [], []
#     for i in range(len(data) - sequence_length - predict_length + 1):
#         X.append(data[i:i + sequence_length])  
#         y.append(data[i + sequence_length:i + sequence_length + predict_length, -1])  
#     return np.array(X), np.array(y)


# sequence_length = 100
# predict_length = 50

# X_train, y_train = prepare_lstm_data(train_set, sequence_length, predict_length)
# X_test, y_test = prepare_lstm_data(test_set, sequence_length, predict_length)

# # Convert to NumPy arrays

# X_train, y_train = np.array(X_train), np.array(y_train)
# X_test, y_test = np.array(X_test), np.array(y_test)



# # Step 4: Build Model
# model = build_model(input_shape=(sequence_length, X_train.shape[2]))  # Adjust shape automatically

# # Step 5: Train Model
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# history = model.fit(
#     X_train, y_train,
#     validation_split=0.2,
#     epochs=50,
#     batch_size=32,
#     callbacks=[early_stopping],
#     verbose=1
# )

# # Step 6: Evaluate Model
# test_loss = model.evaluate(X_test, y_test)
# print(f"Test Loss: {test_loss}")

# # Step 7: Save Model
# model.save('bitcoin_prediction_model.h5')

# # Step 8: Optional - Make Predictions
# predictions = model.predict(X_test[:5])  # Predict the first 5 sequences in the test set
# print("Predictions for the first 5 sequences:")
# print(predictions)

import pandas as pd
from splitdata import split_data
from normalize_ohlcv import normalize_ohlcv
from deeplearningmodel import build_model
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load Data
file_path = 'https://raw.githubusercontent.com/koreakimbap/bitcoin_prediction_ai/main/btc_with_normalized_indicators.csv'
data = pd.read_csv(file_path)

traindata = data.select_dtypes(include=[np.number])
traindata = traindata.fillna(0)

# Step 2: Split Data (Train/Test)
train_set, test_set = split_data(traindata, train_ratio=0.8, sequence_length=100)

# Step 3: Reduce Size of Data (Use only a subset)
train_set = train_set[:10000]  # Use the first 10,000 rows for training data
test_set = test_set[:2000]     # Use the first 2,000 rows for testing data

# Step 4: Prepare Data for LSTM (Create X and y)
def prepare_lstm_data(data, sequence_length, predict_length):
    X, y = [], []
    for i in range(len(data) - sequence_length - predict_length + 1):
        X.append(data[i:i + sequence_length])  
        y.append(data[i + sequence_length:i + sequence_length + predict_length, -1])  
    return np.array(X), np.array(y)

sequence_length = 100
predict_length = 50

X_train, y_train = prepare_lstm_data(train_set, sequence_length, predict_length)
X_test, y_test = prepare_lstm_data(test_set, sequence_length, predict_length)

# 오류 ㅅㅂ
# X_train과 X_test를 5D 텐서로 변환
X_train = X_train.reshape(X_train.shape[0], sequence_length, 100, 1, 1)  # sequence_length, height, width, channels
X_test = X_test.reshape(X_test.shape[0], sequence_length, 100, 1, 1)

# Step 5: Build Model
model = build_model(input_shape=(sequence_length, X_train.shape[2]))  # Adjust shape automatically

# Step 6: Train Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Step 7: Evaluate Model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Step 8: Save Model
model.save('bitcoin_prediction_model.h5')

# Step 9: Optional - Make Predictions
predictions = model.predict(X_test[:5])  # Predict the first 5 sequences in the test set
print("Predictions for the first 5 sequences:")
print(predictions)
