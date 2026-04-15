import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 加载向量和标签
train_vectors = np.load('train_vectors.npy')
train_labels = np.load('train_labels.npy')
test_vectors = np.load('test_vectors.npy')

# 训练Logistic Regression模型
print("训练Logistic Regression模型...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(train_vectors, train_labels)

# 加载测试数据
test_df = pd.read_csv('processed_test.csv')

# 预测测试集
print("预测测试集...")
y_pred = model.predict(test_vectors)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': y_pred
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)

print("提交文件已生成: submission.csv")
print(f"提交文件大小: {len(submission)}")
