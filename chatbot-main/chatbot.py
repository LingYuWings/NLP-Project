import tensorflow as tf
import numpy as np

# 假设有一些对话数据
input_texts = ["Hi", "How are you?", "What's your name?", "Goodbye"]
target_texts = ["Hello", "I'm fine, thanks!", "I'm a chatbot.", "See you later"]

# 构建词汇表
input_vocab = set(" ".join(input_texts).split())
target_vocab = set(" ".join(target_texts).split())
input_vocab_size = len(input_vocab)
target_vocab_size = len(target_vocab)

# 构建单词到索引的映射
input_token_index = dict([(word, i) for i, word in enumerate(input_vocab)])
target_token_index = dict([(word, i) for i, word in enumerate(target_vocab)])

# 构建训练数据
encoder_input_data = []
decoder_input_data = []
decoder_target_data = []

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    encoder_input_data.append([input_token_index[word] for word in input_text.split()])
    decoder_input_data.append([target_token_index[word] for word in target_text.split()])
    decoder_target_data.append([target_token_index[word] for word in target_text.split()[1:]])

# 填充序列
max_encoder_seq_length = max([len(seq) for seq in encoder_input_data])
max_decoder_seq_length = max([len(seq) for seq in decoder_input_data])

encoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_data, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_data, maxlen=max_decoder_seq_length, padding='post')
decoder_target_data = tf.keras.preprocessing.sequence.pad_sequences(decoder_target_data, maxlen=max_decoder_seq_length, padding='post')

# 构建Seq2Seq模型
embedding_dim = 128
hidden_units = 256

# 编码器
encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm, state_h, state_c = tf.keras.layers.LSTM(hidden_units, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# 解码器
decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 定义模型
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1), batch_size=2, epochs=50)

# 进行推断
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(hidden_units,))
decoder_state_input_c = tf.keras.layers.Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_token_index['<START>']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_token_index[sampled_token_index]
        decoded_sentence += ' ' + sampled_word

        if (sampled_word == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence

# 进行测试
for seq_index in range(len(encoder_input_data)):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input:', input_texts[seq_index])
    print('Decoded:', decoded_sentence)
