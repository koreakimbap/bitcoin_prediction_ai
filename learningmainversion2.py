import pandas as pd
from splitdata import split_data
from normalize_ohlcv import normalize_ohlcv
from deeplearningmodel import build_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Step 1: Load Data
file_path = 'btc_with_normalized_indicators.csv'
data = pd.read_csv(file_path)

data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Step 2: Normalize and Split Data
train_set, test_set = split_data(data, train_ratio=0.8, sequence_length=100)

# Step 3: Define Data Generator for Mini-Batch Training
def data_generator(data, sequence_length, predict_length, batch_size):
    num_samples = len(data) - sequence_length - predict_length + 1
    while True:  # Infinite loop for generator
        for i in range(0, num_samples, batch_size):
            X, y = [], []
            for j in range(i, min(i + batch_size, num_samples)):
                X.append(data[j:j + sequence_length])
                y.append(data[j + sequence_length:j + sequence_length + predict_length, -1])  # Target: close
            yield np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Step 4: Prepare Generators
sequence_length = 100
predict_length = 50
batch_size = 32

# Train and Test Generators
# Step 4: Prepare Generators
train_generator = data_generator(train_set, sequence_length, predict_length, batch_size)  
test_generator = data_generator(test_set, sequence_length, predict_length, batch_size)   


# Compute Steps Per Epoch
steps_per_epoch = (len(train_set) - sequence_length - predict_length + 1) // batch_size
validation_steps = (len(test_set) - sequence_length - predict_length + 1) // batch_size

# Step 5: Build Model
model = build_model(input_shape=(sequence_length, 10))  # Automatically adjust input shape

# Step 6: Train Model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps,
    epochs=50,
    callbacks=[early_stopping],
    verbose=1
)

# Step 7: Evaluate Model
test_loss = model.evaluate(test_generator, steps=validation_steps)
print(f"Test Loss: {test_loss}")

# Step 8: Save Model
model.save('bitcoin_prediction_model.h5')
