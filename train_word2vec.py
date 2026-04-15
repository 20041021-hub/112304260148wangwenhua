import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# 读取预处理后的数据
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')

# 将文本转换为单词列表
def tokenize(text):
    return simple_preprocess(text)

# 准备训练数据
train_corpus = [tokenize(review) for review in train_df['processed_review']]
test_corpus = [tokenize(review) for review in test_df['processed_review']]

# 训练Word2Vec模型
print("训练Word2Vec模型...")
model = Word2Vec(
    sentences=train_corpus,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=10
)

# 保存模型
model.save('word2vec_model.model')

print("Word2Vec模型训练完成，已保存为 word2vec_model.model")
print(f"模型词汇表大小: {len(model.wv.key_to_index)}")
