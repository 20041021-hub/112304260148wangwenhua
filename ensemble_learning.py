import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import hstack

def train_model(model, X, y):
    """训练模型"""
    model.fit(X, y)
    return model

def predict_model(model, X):
    """预测模型"""
    return model.predict_proba(X)[:, 1]

def main():
    # 1. 加载数据
    print("步骤1: 加载数据...")
    train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
    test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
    
    # 2. 清洗文本
    print("\n步骤2: 清洗文本...")
    from text_cleaner import clean_text
    train_df['cleaned_review'] = train_df['review'].apply(clean_text)
    test_df['cleaned_review'] = test_df['review'].apply(clean_text)
    
    # 3. 特征工程 - 多种特征组合
    print("\n步骤3: 特征工程...")
    
    # 3.1 TF-IDF特征 (1-4 gram)
    print("  - 提取TF-IDF特征 (1-4 gram)...")
    tfidf_14 = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 4),
        sublinear_tf=True,
        min_df=5,
        max_df=0.8
    )
    X_train_tfidf_14 = tfidf_14.fit_transform(train_df['cleaned_review'])
    X_test_tfidf_14 = tfidf_14.transform(test_df['cleaned_review'])
    
    # 3.2 TF-IDF特征 (2-3 gram)
    print("  - 提取TF-IDF特征 (2-3 gram)...")
    tfidf_23 = TfidfVectorizer(
        max_features=30000,
        ngram_range=(2, 3),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    )
    X_train_tfidf_23 = tfidf_23.fit_transform(train_df['cleaned_review'])
    X_test_tfidf_23 = tfidf_23.transform(test_df['cleaned_review'])
    
    # 3.3 CountVectorizer特征 (1-3 gram)
    print("  - 提取CountVectorizer特征 (1-3 gram)...")
    count_13 = CountVectorizer(
        max_features=20000,
        ngram_range=(1, 3)
    )
    X_train_count_13 = count_13.fit_transform(train_df['cleaned_review'])
    X_test_count_13 = count_13.transform(test_df['cleaned_review'])
    
    # 4. 组合特征
    print("\n步骤4: 组合特征...")
    X_train_combined = hstack([X_train_tfidf_14, X_train_tfidf_23, X_train_count_13])
    X_test_combined = hstack([X_test_tfidf_14, X_test_tfidf_23, X_test_count_13])
    y_train = train_df['sentiment'].values
    
    # 5. 划分验证集
    print("\n步骤5: 划分验证集...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_combined, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 6. 训练多个模型
    print("\n步骤6: 训练多个模型...")
    
    # 模型1: Logistic Regression with C=2.0
    print("  - 训练模型1: Logistic Regression (C=2.0)...")
    model1 = LogisticRegression(
        C=2.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
    model1 = train_model(model1, X_train_combined, y_train)
    
    # 模型2: Logistic Regression with C=5.0
    print("  - 训练模型2: Logistic Regression (C=5.0)...")
    model2 = LogisticRegression(
        C=5.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
    model2 = train_model(model2, X_train_combined, y_train)
    
    # 模型3: Logistic Regression with C=1.0 (更正则化)
    print("  - 训练模型3: Logistic Regression (C=1.0)...")
    model3 = LogisticRegression(
        C=1.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
    model3 = train_model(model3, X_train_combined, y_train)
    
    # 7. 集成预测
    print("\n步骤7: 集成预测...")
    
    # 在验证集上评估
    print("  - 在验证集上评估...")
    y_pred_proba1 = predict_model(model1, X_val)
    y_pred_proba2 = predict_model(model2, X_val)
    y_pred_proba3 = predict_model(model3, X_val)
    
    # 简单平均集成
    y_pred_proba_ensemble = (y_pred_proba1 + y_pred_proba2 + y_pred_proba3) / 3
    
    # 评估集成结果
    auc = roc_auc_score(y_val, y_pred_proba_ensemble)
    y_pred = (y_pred_proba_ensemble >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"  集成模型验证集AUC: {auc:.4f}")
    print(f"  集成模型验证集准确率: {accuracy:.4f}")
    
    # 8. 交叉验证
    print("\n步骤8: 交叉验证...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    accuracy_scores = []
    
    for train_idx, val_idx in skf.split(X_train_combined, y_train):
        X_train_cv = X_train_combined[train_idx]
        y_train_cv = y_train[train_idx]
        X_val_cv = X_train_combined[val_idx]
        y_val_cv = y_train[val_idx]
        
        # 训练三个模型
        m1 = LogisticRegression(C=2.0, solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)
        m2 = LogisticRegression(C=5.0, solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)
        m3 = LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)
        
        m1.fit(X_train_cv, y_train_cv)
        m2.fit(X_train_cv, y_train_cv)
        m3.fit(X_train_cv, y_train_cv)
        
        # 集成预测
        p1 = m1.predict_proba(X_val_cv)[:, 1]
        p2 = m2.predict_proba(X_val_cv)[:, 1]
        p3 = m3.predict_proba(X_val_cv)[:, 1]
        p_ensemble = (p1 + p2 + p3) / 3
        
        # 评估
        auc = roc_auc_score(y_val_cv, p_ensemble)
        acc = accuracy_score(y_val_cv, (p_ensemble >= 0.5).astype(int))
        
        auc_scores.append(auc)
        accuracy_scores.append(acc)
    
    print(f"交叉验证AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"交叉验证准确率: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    
    # 9. 预测测试集
    print("\n步骤9: 预测测试集...")
    test_pred_proba1 = predict_model(model1, X_test_combined)
    test_pred_proba2 = predict_model(model2, X_test_combined)
    test_pred_proba3 = predict_model(model3, X_test_combined)
    
    # 集成预测
    test_pred_proba_ensemble = (test_pred_proba1 + test_pred_proba2 + test_pred_proba3) / 3
    test_pred = (test_pred_proba_ensemble >= 0.5).astype(int)
    
    # 10. 创建提交文件
    print("\n步骤10: 创建提交文件...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件已生成: submission.csv")
    print(f"提交文件大小: {len(submission)}")

if __name__ == "__main__":
    main()
