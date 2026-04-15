import pandas as pd
from gensim.models import Word2Vec

# 读取预处理后的数据
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')

# 将文本转换为单词列表
def tokenize(text):
    return text.split()

# 准备训练数据
train_corpus = [tokenize(review) for review in train_df['processed_review']]

# 训练Word2Vec模型
print("训练Word2Vec模型...")
model = Word2Vec(
    sentences=train_corpus,
    vector_size=300,  # 增加向量维度
    window=5,  # 窗口大小
    min_count=5,  # 最小词频
    workers=4,
    epochs=30,  # 增加训练轮数
    sg=1,  # 使用skip-gram模型
    negative=10,  # 负采样
    hs=0  # 不使用层次softmax
)

# 保存模型
model.save('word2vec_model.model')

# 保存词向量，方便后续直接加载
model.wv.save('word2vec_vectors.kv')

print("Word2Vec模型训练完成，已保存为 word2vec_model.model")
print("词向量已保存为 word2vec_vectors.kv")
print(f"模型词汇表大小: {len(model.wv.key_to_index)}")
