import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

# 加载词向量
print("加载词向量...")
word_vectors = KeyedVectors.load('word2vec_vectors.kv')
vector_size = word_vectors.vector_size

# 读取预处理后的数据
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')

# 将文本转换为句向量（使用简单的均值embedding）
def text_to_vector(text):
    words = text.split()
    vectors = []
    for word in words:
        if word in word_vectors:
            vectors.append(word_vectors[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# 转换训练集文本为向量
print("转换训练集文本为向量...")
train_vectors = [text_to_vector(review) for review in train_df['processed_review']]
train_vectors = np.array(train_vectors)

# 转换测试集文本为向量
print("转换测试集文本为向量...")
test_vectors = [text_to_vector(review) for review in test_df['processed_review']]
test_vectors = np.array(test_vectors)

# 保存向量
np.save('train_vectors.npy', train_vectors)
np.save('test_vectors.npy', test_vectors)

# 保存训练集标签
np.save('train_labels.npy', train_df['sentiment'].values)

print("文本向量化完成，向量已保存为 train_vectors.npy 和 test_vectors.npy")
print(f"训练集向量形状: {train_vectors.shape}")
print(f"测试集向量形状: {test_vectors.shape}")
