# -*- coding: utf-8 -*-
"""chatbot1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qLNjKWfHC7Lq1WLiBCIWzqHuH0iceHch
"""



import pandas as pd
import numpy as np

# 读取txt文件为DataFrame
file_path = 'formatted_movie_lines.txt'
data = pd.read_csv(file_path, sep='\t', header=None, names=['speaker1', 'speaker2'])

# 随机选取1%的数据
percentage = 0.1
sampled_data = data.sample(frac=percentage, random_state=42)
data=sampled_data

# 可选：查看随机选取的数据
print(sampled_data.head())
len(sampled_data)

input_sentences = data['speaker1'].tolist()
output_sentences = data['speaker2'].tolist()

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# 分词函数
def tokenize_sentences(sentences):
    tokenized = [word_tokenize(sentence.lower()) for sentence in sentences]
    return tokenized

input_tokenized = tokenize_sentences(input_sentences)
output_tokenized = tokenize_sentences(output_sentences)

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_tokenized + output_tokenized)

input_sequences = tokenizer.texts_to_sequences(input_tokenized)
output_sequences = tokenizer.texts_to_sequences(output_tokenized)

vocab_size = len(tokenizer.word_index) + 1

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_tokenized + output_tokenized)

input_sequences = tokenizer.texts_to_sequences(input_tokenized)
output_sequences = tokenizer.texts_to_sequences(output_tokenized)

vocab_size = len(tokenizer.word_index) + 1

from tensorflow.keras.preprocessing.sequence import pad_sequences

max_sequence_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in output_sequences))

input_padded = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
output_padded = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='pre')

X = np.array(input_padded)
y = np.array(output_padded)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

embedding_dim = 64
hidden_units = 128

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),
    LSTM(hidden_units, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, y, batch_size=32, epochs=1, validation_split=0.1)

# 保存模型
model.save('my_chatbot_model.h5')

# from tensorflow.keras.models import load_model

# # 加载模型
# loaded_model = model

def generate_reply(user_input, model, tokenizer, max_sequence_length):
    input_sequence = tokenizer.texts_to_sequences([user_input])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='pre')

    predicted_sequence = model.predict(input_padded)
    predicted_index = np.argmax(predicted_sequence, axis=-1)

    # 将预测的整数序列转换回文本
    predicted_text = tokenizer.sequences_to_texts(predicted_index)

    return predicted_text

# 用户输入
user_input = "Nah."

# 生成回复
reply = generate_reply(user_input, model, tokenizer, 443)
print("Chatbot:", reply)