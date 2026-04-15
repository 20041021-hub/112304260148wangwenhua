import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# 加载向量和标签
train_vectors = np.load('train_vectors.npy')
train_labels = np.load('train_labels.npy')

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    train_vectors, train_labels, test_size=0.2, random_state=42
)

# 调优逻辑回归模型
print("调优逻辑回归模型...")
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [1000, 2000]
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid=param_grid,
    cv=5,  # 增加交叉验证折数
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"最佳逻辑回归参数: {grid_search.best_params_}")
print(f"最佳交叉验证AUC: {grid_search.best_score_:.4f}")

# 评估最佳逻辑回归模型
best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
test_auc = roc_auc_score(y_val, y_pred_proba)
print(f"测试集AUC: {test_auc:.4f}")

# 计算准确率
y_pred = best_model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"测试集准确率: {acc:.4f}")

# 保存最佳模型
import joblib
joblib.dump(best_model, 'best_model.joblib')
print("最佳模型已保存为 best_model.joblib")
