import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 加载Word2Vec模型
model = Word2Vec.load('word2vec_model.model')

# 将文本转换为句向量（单词向量的平均值）
def text_to_vector(text):
    words = simple_preprocess(text)
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 读取预处理后的数据
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')

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
