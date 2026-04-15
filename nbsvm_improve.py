import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import hstack, csr_matrix

def tokenize(sentence, ngram_range=(1, 1)):
    """分词函数，支持n-gram"""
    words = sentence.split()
    tokens = []
    for n in range(ngram_range[0], ngram_range[1]+1):
        for i in range(len(words) - n + 1):
            tokens.append(" ".join(words[i:i+n]))
    return tokens

def compute_ratio(train_texts, train_labels, alpha=1.0):
    """计算每个词的情感倾向比率"""
    # 分离正面和负面文本
    pos_texts = train_texts[train_labels == 1]
    neg_texts = train_texts[train_labels == 0]
    
    # 计算词频
    pos_counts = {}
    neg_counts = {}
    
    for text in pos_texts:
        tokens = tokenize(text, ngram_range=(1, 4))
        for token in tokens:
            pos_counts[token] = pos_counts.get(token, 0) + 1
    
    for text in neg_texts:
        tokens = tokenize(text, ngram_range=(1, 4))
        for token in tokens:
            neg_counts[token] = neg_counts.get(token, 0) + 1
    
    # 计算所有词的集合
    all_tokens = list(set(pos_counts.keys()) | set(neg_counts.keys()))
    token_to_idx = {token: i for i, token in enumerate(all_tokens)}
    vocab_size = len(all_tokens)
    
    # 计算比率
    p = np.ones(vocab_size) * alpha
    q = np.ones(vocab_size) * alpha
    
    for token, idx in token_to_idx.items():
        p[idx] += pos_counts.get(token, 0)
        q[idx] += neg_counts.get(token, 0)
    
    p /= p.sum()
    q /= q.sum()
    
    # 计算比率的对数
    ratio = np.log(p / q)
    
    return token_to_idx, ratio

def create_nbsvm_features(texts, token_to_idx, ratio):
    """创建NBSVM特征"""
    features = []
    for text in texts:
        tokens = tokenize(text, ngram_range=(1, 4))
        token_indices = [token_to_idx.get(token, -1) for token in tokens]
        token_indices = [idx for idx in token_indices if idx != -1]
        
        # 计算特征向量
        feature = np.zeros(len(ratio))
        for idx in token_indices:
            feature[idx] += ratio[idx]
        
        features.append(feature)
    
    return np.array(features)

def main():
    # 1. 加载数据
    print("步骤1: 加载数据...")
    train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
    test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
    
    # 2. 清洗文本（使用之前的clean_text函数）
    print("\n步骤2: 清洗文本...")
    from text_cleaner import clean_text
    train_df['cleaned_review'] = train_df['review'].apply(clean_text)
    test_df['cleaned_review'] = test_df['review'].apply(clean_text)
    
    # 3. 计算NBSVM特征
    print("\n步骤3: 计算NBSVM特征...")
    train_texts = train_df['cleaned_review'].values
    train_labels = train_df['sentiment'].values
    
    # 计算比率
    token_to_idx, ratio = compute_ratio(train_texts, train_labels)
    
    # 创建训练集特征
    X_train_nbsvm = create_nbsvm_features(train_texts, token_to_idx, ratio)
    
    # 创建测试集特征
    test_texts = test_df['cleaned_review'].values
    X_test_nbsvm = create_nbsvm_features(test_texts, token_to_idx, ratio)
    
    # 4. 添加TF-IDF特征
    print("\n步骤4: 添加TF-IDF特征...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 4),
        sublinear_tf=True,
        min_df=5,
        max_df=0.8
    )
    
    X_train_tfidf = tfidf.fit_transform(train_df['cleaned_review'])
    X_test_tfidf = tfidf.transform(test_df['cleaned_review'])
    
    # 5. 结合特征
    print("\n步骤5: 结合特征...")
    X_train_combined = hstack([csr_matrix(X_train_nbsvm), X_train_tfidf])
    X_test_combined = hstack([csr_matrix(X_test_nbsvm), X_test_tfidf])
    
    # 6. 划分验证集
    print("\n步骤6: 划分验证集...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_combined, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # 7. 训练模型
    print("\n步骤7: 训练模型...")
    model = LogisticRegression(
        C=5.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
    
    # 8. 交叉验证
    print("\n步骤8: 交叉验证...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    accuracy_scores = []
    
    for train_idx, val_idx in skf.split(X_train_combined, train_labels):
        X_train_cv = X_train_combined[train_idx]
        y_train_cv = train_labels[train_idx]
        X_val_cv = X_train_combined[val_idx]
        y_val_cv = train_labels[val_idx]
        
        model.fit(X_train_cv, y_train_cv)
        
        y_pred_proba = model.predict_proba(X_val_cv)[:, 1]
        y_pred = model.predict(X_val_cv)
        
        auc = roc_auc_score(y_val_cv, y_pred_proba)
        accuracy = accuracy_score(y_val_cv, y_pred)
        
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
    
    print(f"交叉验证AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"交叉验证准确率: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    
    # 9. 训练完整模型
    print("\n步骤9: 训练完整模型...")
    model.fit(X_train_combined, train_labels)
    
    # 10. 评估模型
    print("\n步骤10: 评估模型...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"验证集AUC: {auc:.4f}")
    print(f"验证集准确率: {accuracy:.4f}")
    
    # 11. 预测测试集
    print("\n步骤11: 预测测试集...")
    test_predictions = model.predict(X_test_combined)
    
    # 12. 创建提交文件
    print("\n步骤12: 创建提交文件...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件已生成: submission.csv")
    print(f"提交文件大小: {len(submission)}")

if __name__ == "__main__":
    main()
