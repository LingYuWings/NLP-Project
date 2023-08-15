import tensorflow as tf
import numpy as np

# 准备数据
# 示例数据，格式为（输入句子，目标回复）
data = [
    ("你好", "你好啊"),
    ("有什么好看的电影推荐吗？", "我推荐你看《肖申克的救赎》。"),
    # 添加更多数据...
]

# 构建词汇表
vocab = set()
for input_text, target_text in data:
    vocab.update(input_text.split())
    vocab.update(target_text.split())
vocab = sorted(vocab)
vocab_size = len(vocab)

# 构建单词到索引的映射
word2idx = {word: idx for idx, word in enumerate(vocab)}

# 将文本转换为索引序列
def text_to_indices(text):
    return [word2idx[word] for word in text.split()]

# 将输入和目标回复转换为索引序列
input_data = [text_to_indices(input_text) for input_text, _ in data]
target_data = [text_to_indices(target_text) for _, target_text in data]

# 填充序列以使其具有相同的长度
max_sequence_length = max(max(map(len, input_data)), max(map(len, target_data)))
input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_sequence_length, padding='pre')
target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=max_sequence_length, padding='pre')

# 构建模型
embedding_dim = 128
hidden_units = 256

encoder_inputs = tf.keras.layers.Input(shape=(max_sequence_length,))
encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_rnn = tf.keras.layers.LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c = encoder_rnn(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(max_sequence_length,))
decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_rnn = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 定义模型
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit([input_data, target_data[:, :-1]], target_data[:, 1:], batch_size=64, epochs=50)

# 生成回复
# 生成回复
def generate_reply(input_text):
    input_indices = np.array([text_to_indices(input_text)])
    # 加载已训练的编码器部分权重
    encoder_rnn.set_weights(model.get_layer('lstm')(encoder_inputs)[1])
    initial_state = encoder_rnn.predict(input_indices)
    target_seq = np.array([[word2idx['<start>']]])
    
    reply = []
    for _ in range(max_sequence_length):
        decoder_output, state_h, state_c = decoder_rnn(decoder_embedding(target_seq), initial_state=initial_state)
        decoder_output = decoder_dense(decoder_output)
        predicted_word_idx = np.argmax(decoder_output[0, -1, :])
        if predicted_word_idx == word2idx['<end>']:
            break
        reply.append(vocab[predicted_word_idx])
        target_seq = np.array([[predicted_word_idx]])
        initial_state = [state_h, state_c]
    
    return ' '.join(reply)


# 测试生成回复
input_text = "你好"
generated_reply = generate_reply(input_text)
print(f"Input: {input_text}")
print(f"Generated reply: {generated_reply}")
