import json
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Load and preprocess the data
with open("idmanual.json") as f:  # Replace "your_data.json" with your JSON file path
    data = json.load(f)

texts = []
labels = []

for entry in data:
    texts.append(entry['description'])
    labels.append(entry['class_id'])

texts = np.array(texts)
labels = np.array(labels)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
max_seq_length = 100  # Replace with your desired sequence length
sequences = pad_sequences(sequences, maxlen=max_seq_length)

# Convert labels to integer format
label_to_id = {label: idx for idx, label in enumerate(np.unique(labels))}
id_to_label = {idx: label for label, idx in label_to_id.items()}
labels = np.array([label_to_id[label] for label in labels])

# Define early stopping callback
early_stopping = EarlyStopping(patience=2, restore_best_weights=True)

# Define the GRU model
embedding_dim = 100  # Replace with your desired embedding dimension
num_classes = len(np.unique(labels))


def train_model():
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_seq_length))
    model.add(GRU(128))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define early stopping callback
    early_stopping = EarlyStopping(patience=3, restore_best_weights=True)

    # Train the model
    model.fit(sequences, labels, epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping])
    # Evaluate the model on the entire dataset
    loss, accuracy = model.evaluate(sequences, labels)

    # Save the model
    model.save('trained_model.h5')
    return {"Accuracy": accuracy, "Message": "Model saved successfully."}


def predict(user_input):
    # Load the saved model
    loaded_model = load_model('trained_model.h5')
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_sequence = pad_sequences(user_sequence, maxlen=max_seq_length)
    # Make predictions using the loaded model
    prediction = loaded_model.predict(user_sequence)
    predicted_class_id = np.argmax(prediction)
    predicted_class_label = id_to_label[predicted_class_id]
    return f'Predicted Class: {predicted_class_label}'
