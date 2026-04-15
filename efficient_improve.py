import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from scipy.sparse import hstack
import joblib
import re
import nltk
from nltk.corpus import stopwords

# 下载停用词
nltk.download('stopwords')

# 加载英文停用词
stop_words = set(stopwords.words('english'))

# 改进的预处理函数（保留情感标点）
def improved_preprocess(text):
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 小写化
    text = text.lower()
    # 保留情感标点，移除其他标点
    text = re.sub(r'[.,;:]', ' ', text)
    # 多个空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 停用词处理
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# 1. 重新预处理数据
print("重新预处理数据...")
train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')

train_df['processed_review'] = train_df['review'].apply(improved_preprocess)
test_df['processed_review'] = test_df['review'].apply(improved_preprocess)

# 2. 检查类别分布
print("检查类别分布...")
sentiment_counts = train_df['sentiment'].value_counts()
print(f"类别分布: {sentiment_counts}")
print(f"正样本比例: {sentiment_counts[1]/len(train_df):.4f}")

# 3. 加载Word2Vec向量
train_vectors = np.load('train_vectors.npy')
test_vectors = np.load('test_vectors.npy')
train_labels = np.load('train_labels.npy')

# 4. 添加TF-IDF特征
print("提取TF-IDF特征...")
tfidf = TfidfVectorizer(max_features=5000)
train_tfidf = tfidf.fit_transform(train_df['processed_review'])
test_tfidf = tfidf.transform(test_df['processed_review'])

# 5. 划分训练集和验证集
indices = np.arange(len(train_vectors))
X_train_indices, X_val_indices, y_train, y_val = train_test_split(
    indices, train_labels, test_size=0.2, random_state=42
)
X_train = train_vectors[X_train_indices]
X_val = train_vectors[X_val_indices]
X_train_tfidf = train_tfidf[X_train_indices]
X_val_tfidf = train_tfidf[X_val_indices]

# 6. 结合特征
print("结合特征...")
X_train_combined = hstack([X_train, X_train_tfidf])
X_val_combined = hstack([X_val, X_val_tfidf])
test_combined = hstack([test_vectors, test_tfidf])

# 7. 训练逻辑回归模型（使用类别权重）
print("训练逻辑回归模型...")
model = LogisticRegression(
    random_state=42, 
    max_iter=2000, 
    class_weight='balanced',
    C=1.0,
    solver='liblinear'
)

# 训练模型
model.fit(X_train_combined, y_train)

# 8. 预测验证集
print("预测验证集...")
y_pred_proba = model.predict_proba(X_val_combined)[:, 1]

# 计算AUC
auc = roc_auc_score(y_val, y_pred_proba)
print(f"验证集AUC: {auc:.4f}")

# 9. 寻找最优阈值
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

# 10. 保存模型
joblib.dump(model, 'optimized_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
print("优化模型已保存")

# 11. 预测测试集
print("预测测试集...")
test_pred_proba = model.predict_proba(test_combined)[:, 1]
test_pred = (test_pred_proba >= optimal_threshold).astype(int)

# 12. 创建提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': test_pred
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print("提交文件已生成: submission.csv")
print(f"提交文件大小: {len(submission)}")
