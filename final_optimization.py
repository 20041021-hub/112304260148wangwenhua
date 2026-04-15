import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import hstack

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
    
    # 3. 特征工程 - 最佳特征组合
    print("\n步骤3: 特征工程...")
    
    # 3.1 最佳TF-IDF特征 (1-4 gram)
    print("  - 提取TF-IDF特征 (1-4 gram)...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 4),
        sublinear_tf=True,
        min_df=5,
        max_df=0.8
    )
    X_train = tfidf.fit_transform(train_df['cleaned_review'])
    X_test = tfidf.transform(test_df['cleaned_review'])
    y_train = train_df['sentiment'].values
    
    # 4. 划分验证集
    print("\n步骤4: 划分验证集...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 5. 精细调优模型参数
    print("\n步骤5: 精细调优模型参数...")
    
    # 更精细的C值范围
    C_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    best_auc = 0
    best_C = None
    
    for C in C_values:
        print(f"  - 测试C={C}...")
        model = LogisticRegression(
            C=C,
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        )
        
        # 5折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_idx, val_idx in skf.split(X_train_split, y_train_split):
            X_train_cv = X_train_split[train_idx]
            y_train_cv = y_train_split[train_idx]
            X_val_cv = X_train_split[val_idx]
            y_val_cv = y_train_split[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_proba = model.predict_proba(X_val_cv)[:, 1]
            auc = roc_auc_score(y_val_cv, y_pred_proba)
            auc_scores.append(auc)
        
        mean_auc = np.mean(auc_scores)
        print(f"    5折交叉验证AUC: {mean_auc:.4f}")
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_C = C
    
    print(f"\n最佳C值: {best_C}")
    print(f"最佳交叉验证AUC: {best_auc:.4f}")
    
    # 6. 训练最佳模型
    print("\n步骤6: 训练最佳模型...")
    best_model = LogisticRegression(
        C=best_C,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
    best_model.fit(X_train, y_train)
    
    # 7. 评估模型
    print("\n步骤7: 评估模型...")
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred = best_model.predict(X_val)
    
    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"验证集AUC: {auc:.4f}")
    print(f"验证集准确率: {accuracy:.4f}")
    
    # 8. 最终交叉验证
    print("\n步骤8: 最终5折交叉验证...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    accuracy_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_val_cv = X_train[val_idx]
        y_val_cv = y_train[val_idx]
        
        model = LogisticRegression(
            C=best_C,
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        )
        model.fit(X_train_cv, y_train_cv)
        
        y_pred_proba = model.predict_proba(X_val_cv)[:, 1]
        y_pred = model.predict(X_val_cv)
        
        auc = roc_auc_score(y_val_cv, y_pred_proba)
        accuracy = accuracy_score(y_val_cv, y_pred)
        
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
    
    print(f"5折交叉验证AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"5折交叉验证准确率: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    
    # 9. 预测测试集
    print("\n步骤9: 预测测试集...")
    # 输出概率而不是硬标签
    test_predictions = best_model.predict_proba(X_test)[:, 1]
    
    # 10. 创建提交文件
    print("\n步骤10: 创建提交文件...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件已生成: submission.csv")
    print(f"提交文件大小: {len(submission)}")
    # 显示前5行，确认是概率值
    print("\n提交文件前5行:")
    print(submission.head())

if __name__ == "__main__":
    main()
