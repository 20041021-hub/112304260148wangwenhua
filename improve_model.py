import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from scipy.sparse import hstack
import joblib

# 加载数据
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')
train_vectors = np.load('train_vectors.npy')
test_vectors = np.load('test_vectors.npy')
train_labels = np.load('train_labels.npy')

# 划分训练集和验证集
indices = np.arange(len(train_vectors))
X_train_indices, X_val_indices, y_train, y_val = train_test_split(
    indices, train_labels, test_size=0.2, random_state=42
)
X_train = train_vectors[X_train_indices]
X_val = train_vectors[X_val_indices]

# 添加TF-IDF特征
print("提取TF-IDF特征...")
tfidf = TfidfVectorizer(max_features=5000)
train_tfidf = tfidf.fit_transform(train_df['processed_review'])
test_tfidf = tfidf.transform(test_df['processed_review'])

# 划分TF-IDF特征
X_train_tfidf = train_tfidf[X_train_indices]
X_val_tfidf = train_tfidf[X_val_indices]

# 结合Word2Vec和TF-IDF特征
print("结合特征...")
X_train_combined = hstack([X_train, X_train_tfidf])
X_val_combined = hstack([X_val, X_val_tfidf])
test_combined = hstack([test_vectors, test_tfidf])

# 训练逻辑回归模型
print("训练逻辑回归模型...")
model = LogisticRegression(random_state=42, max_iter=2000)
model.fit(X_train_combined, y_train)

# 预测验证集
print("预测验证集...")
y_pred_proba = model.predict_proba(X_val_combined)[:, 1]

# 计算AUC
auc = roc_auc_score(y_val, y_pred_proba)
print(f"验证集AUC: {auc:.4f}")

# 寻找最优阈值
print("寻找最优阈值...")
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
# 计算F1分数
f1_scores = 2 * (precision * recall) / (precision + recall)
# 找到最优阈值
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"最优阈值: {optimal_threshold:.4f}")

# 使用最优阈值计算准确率
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
acc_optimal = accuracy_score(y_val, y_pred_optimal)
print(f"使用最优阈值的准确率: {acc_optimal:.4f}")

# 保存模型
joblib.dump(model, 'best_model_improved.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
print("改进后的模型已保存")

# 预测测试集
print("预测测试集...")
test_pred_proba = model.predict_proba(test_combined)[:, 1]
test_pred = (test_pred_proba >= optimal_threshold).astype(int)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_pred
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("提交文件已生成: submission.csv")
print(f"提交文件大小: {len(submission)}")
