import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
data = pd.read_csv(r"C:\Users\Gagan\Desktop\SIH\modified_file.csv")

# Feature and label separation
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode categorical data in features
label_encoder_N = LabelEncoder()
label_encoder_P = LabelEncoder()
label_encoder_K = LabelEncoder()

x[:, 0] = label_encoder_N.fit_transform(x[:, 0])  # Encoding Nitrogen
x[:, 1] = label_encoder_P.fit_transform(x[:, 1])  # Encoding Phosphorus
x[:, 2] = label_encoder_K.fit_transform(x[:, 2])  # Encoding Potassium

# Encode categorical data in labels
onehot_encoder = LabelEncoder()
y = onehot_encoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y)  # One-hot encode the labels

# Save the label encoders
with open('label_encoder_N.pkl', 'wb') as f:
    pickle.dump(label_encoder_N, f)
with open('label_encoder_P.pkl', 'wb') as f:
    pickle.dump(label_encoder_P, f)
with open('label_encoder_K.pkl', 'wb') as f:
    pickle.dump(label_encoder_K, f)

with open('label_encoder_labels.pkl', 'wb') as f:
    pickle.dump(onehot_encoder, f)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)

# Build the ANN model
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_shape=(x_train.shape[1],)))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=y.shape[1], activation='softmax'))  # Number of units = number of classes

# Compile the ANN
ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the ANN
ann.fit(x_train, y_train, batch_size=32, epochs=100)

# Save the model in the recommended format
ann.save('crop_recommendation_model.keras')
