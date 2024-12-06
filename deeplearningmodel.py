from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense, Dropout, Input

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),  
        ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False),
        Dropout(0.2),
        Flatten(),
        Dense(50),  # Predicting the next 50 candles
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    return model


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# def build_model(input_shape):
#     model = Sequential([
#         # Input(shape=(sequence_length, 10)),
#         Input(shape=input_shape),  
#         LSTM(64, return_sequences=True),
#         Dropout(0.2),
#         LSTM(64),
#         Dropout(0.2),
#         Dense(50),  # Predicting the next 50 candles
#     ])
    
#     model.compile(optimizer='adam', loss='mse')
    
#     return model
