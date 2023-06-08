import json
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
# import joblib
from keras.models import load_model

with open('traindata (2).json') as f:
    train_data = json.load(f)

# Remove unwanted classes from the training dataset
unwanted_classes = ['000', '200', 'B','A']
train_data = {key: value for key, value in train_data.items() if key not in unwanted_classes}
# Prepare the training data
train_texts = []
train_labels = []
for key, values in train_data.items():
    train_texts.extend(values)
    train_labels.extend([key] * len(values))
# Step 2: Load and preprocess the testing data
with open('testdata (2).json') as f:
    test_data = json.load(f)
# Remove unwanted classes from the testing dataset
test_data = {key: value for key, value in test_data.items() if key not in unwanted_classes}
# Prepare the testing data
test_texts = []
test_labels = []
for key, values in test_data.items():
    test_texts.extend(values)
    test_labels.extend([key] * len(values))

# Step 3: Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_sequence_length = max(len(sequence) for sequence in train_sequences)
train_padded = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

label_to_id = {label: idx for idx, label in enumerate(set(train_labels))}
train_encoded = np.array([label_to_id[label] for label in train_labels])
test_encoded = np.array([label_to_id[label] for label in test_labels])



def train_model():
    # Step 4: Build the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length))
    model.add(GRU(units=128))
    model.add(Dense(units=len(label_to_id), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_padded, train_encoded, epochs=10, batch_size=32)
    # Save the model
    model.save('GRUtrinedmodel')
    return {"Message": "Model saved successfully."}


def predict(user_input):
    # Load the saved model
    loaded_model = load_model('gru_model.h5')
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_padded = pad_sequences(user_sequence, maxlen=max_sequence_length)

    # Make predictions using the loaded model
    user_prediction = loaded_model.predict(user_padded)
    user_prediction = np.argmax(user_prediction, axis=1)[0]

    # Convert the predicted class index back to the label
    id_to_label = {v: k for k, v in label_to_id.items()}
    predicted_label = id_to_label[user_prediction]
    return f'Predicted Class: {predicted_label}'
