import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load and preprocess data
def load_data(data_dir):
    data_x = []
    data_y = []
    label2index = {'entertainment': 0, 'health': 1, 'politics': 2}
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if not os.path.isdir(category_dir):
            continue
        category_idx = label2index.get(category, -1)
        if category_idx == -1:
            continue
        for file_path in glob.glob(os.path.join(category_dir, '*.txt')):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data_x.append(content)
                data_y.append(category)  # Store actual labels, not indices
    return data_x, data_y

data_dir = "./news300"
data_x, data_y = load_data(data_dir)

num_classes = len(set(data_y))
label2index = {'entertainment': 0, 'health': 1, 'politics': 2}

# Split the data into training and validation sets
train_x, validate_x, train_y, validate_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)

# Helper function to map labels to indices and handle unknown labels
def map_label_to_index(label):
    return label2index.get(label, -1)

# Convert labels to numbers and create datasets
train_processed_y = np.array([map_label_to_index(label) for label in train_y])
validate_processed_y = np.array([map_label_to_index(label) for label in validate_y])

# Tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_encodings = tokenizer(train_x, truncation=True, padding=True)
validate_encodings = tokenizer(validate_x, truncation=True, padding=True)

# Convert encodings to dataset format
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_processed_y
)).shuffle(1000).batch(8)

validate_dataset = tf.data.Dataset.from_tensor_slices((
    dict(validate_encodings),
    validate_processed_y
)).batch(8)

# Build the model using TFBertForSequenceClassification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Compile and train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1.0)  # Add clipnorm parameter
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_dataset, epochs=3, validation_data=validate_dataset)

# Evaluation
validation_loss, validation_accuracy = model.evaluate(validate_dataset)
print(f'Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}')

# Inference
funny_strings = ["A clown performed at the circus.",
                 "Eat fruits for better health.",
                 "The senator delivered a powerful speech."]

predicted_classes = ['entertainment', 'health', 'politics']

for text, expected_class in zip(funny_strings, predicted_classes):
    inputs = tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors='tf')
    predictions = model(inputs)[0]
    predicted_class_idx = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_class = list(label2index.keys())[list(label2index.values()).index(predicted_class_idx)]
    print(f"Text: {text}")
    print(f"Predicted Class: {predicted_class}, Expected Class: {expected_class}")
