from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def create_logistic_regression(C=5.0, solver='liblinear', class_weight='balanced'):
    """创建逻辑回归模型"""
    return LogisticRegression(
        C=C,
        solver=solver,
        class_weight=class_weight,
        random_state=42,
        max_iter=2000
    )

def train_model(model, X_train, y_train):
    """训练模型"""
    model.fit(X_train, y_train)
    return model

def cross_validate_model(model, X, y, n_splits=3):
    """交叉验证模型"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    print(f"交叉验证AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"交叉验证准确率: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    
    return np.mean(auc_scores), np.mean(accuracy_scores)

def evaluate_model(model, X_val, y_val):
    """评估模型"""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"验证集AUC: {auc:.4f}")
    print(f"验证集准确率: {accuracy:.4f}")
    
    return auc, accuracy
