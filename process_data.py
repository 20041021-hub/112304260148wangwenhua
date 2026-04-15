import pandas as pd
from preprocess import preprocess_text

# 读取训练集
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')

# 读取测试集
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

# 预处理训练集文本
print("预处理训练集...")
train_df['processed_review'] = train_df['review'].apply(preprocess_text)

# 预处理测试集文本
print("预处理测试集...")
test_df['processed_review'] = test_df['review'].apply(preprocess_text)

# 保存预处理后的数据
train_df.to_csv('processed_train.csv', index=False)
test_df.to_csv('processed_test.csv', index=False)

print("预处理完成，数据已保存为 processed_train.csv 和 processed_test.csv")
print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")
