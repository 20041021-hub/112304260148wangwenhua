import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import hstack, csr_matrix
import re
import nltk
from nltk.corpus import stopwords

# 下载停用词
nltk.download('stopwords')

# 加载英文停用词
stop_words = set(stopwords.words('english'))

# 1. 文本预处理
def clean_text(text):
    """完整的文本清洗流程"""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 小写化
    text = text.lower()
    # 移除标点符号
    text = re.sub(r'[\W_]', ' ', text)
    # 多个空格替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 停用词处理
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    # 否定词处理
    negation_words = {'not', 'no', 'never', 'nor', "n't", 'cannot'}
    processed_words = []
    negation = False
    for word in filtered_words:
        if word in negation_words:
            negation = True
            processed_words.append(word)
        else:
            if negation:
                processed_words.append(word + '_NEG')
                negation = False
            else:
                processed_words.append(word)
    return ' '.join(processed_words)

# 2. NBSVM实现
def compute_ratio(train_texts, train_labels, alpha=1.0):
    """计算每个词的情感倾向比率"""
    # 分离正面和负面文本
    pos_texts = train_texts[train_labels == 1]
    neg_texts = train_texts[train_labels == 0]
    
    # 计算词频
    pos_counts = {}
    neg_counts = {}
    
    for text in pos_texts:
        words = text.split()
        for word in words:
            pos_counts[word] = pos_counts.get(word, 0) + 1
    
    for text in neg_texts:
        words = text.split()
        for word in words:
            neg_counts[word] = neg_counts.get(word, 0) + 1
    
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
    # 限制词汇表大小，只保留前20000个最常见的词
    if len(ratio) > 20000:
        # 按绝对值排序，保留最重要的特征
        top_indices = np.argsort(np.abs(ratio))[-20000:]
        # 创建旧索引到新索引的映射
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(top_indices)}
        ratio_filtered = ratio[top_indices]
    else:
        old_to_new_idx = {old_idx: old_idx for old_idx in range(len(ratio))}
        ratio_filtered = ratio
    
    features = []
    for text in texts:
        words = text.split()
        # 获取每个词的旧索引
        old_indices = [token_to_idx.get(word, -1) for word in words]
        # 过滤掉不在top_indices中的词，并转换为新索引
        new_indices = [old_to_new_idx[old_idx] for old_idx in old_indices if old_idx != -1 and old_idx in old_to_new_idx]
        
        # 计算特征向量
        feature = np.zeros(len(ratio_filtered))
        for idx in new_indices:
            feature[idx] += ratio_filtered[idx]
        
        features.append(feature)
    
    return np.array(features)

# 3. 主函数
def main():
    # 3.1 加载数据
    print("步骤1: 加载数据...")
    train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
    test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
    # 使用error_bad_lines参数跳过有问题的行
    unlabeled_df = pd.read_csv('unlabeledTrainData.tsv/unlabeledTrainData.tsv', sep='\t', on_bad_lines='skip')
    
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"未标记数据大小: {len(unlabeled_df)}")
    
    # 3.2 清洗文本
    print("\n步骤2: 清洗文本...")
    train_df['cleaned_review'] = train_df['review'].apply(clean_text)
    test_df['cleaned_review'] = test_df['review'].apply(clean_text)
    unlabeled_df['cleaned_review'] = unlabeled_df['review'].apply(clean_text)
    
    # 3.3 合并标记和未标记数据进行特征提取
    print("\n步骤3: 特征工程...")
    all_texts = pd.concat([train_df['cleaned_review'], unlabeled_df['cleaned_review']])
    
    # 3.3.1 TF-IDF特征 (1-4 gram)
    print("  - 提取TF-IDF特征 (1-4 gram)...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 4),
        sublinear_tf=True,
        min_df=5,
        max_df=0.8
    )
    tfidf.fit(all_texts)
    
    X_train_tfidf = tfidf.transform(train_df['cleaned_review'])
    X_test_tfidf = tfidf.transform(test_df['cleaned_review'])
    
    # 3.3.2 CountVectorizer特征 (1-3 gram)
    print("  - 提取CountVectorizer特征 (1-3 gram)...")
    count = CountVectorizer(
        max_features=20000,
        ngram_range=(1, 3)
    )
    count.fit(all_texts)
    
    X_train_count = count.transform(train_df['cleaned_review'])
    X_test_count = count.transform(test_df['cleaned_review'])
    
    # 3.3.3 NBSVM特征
    print("  - 提取NBSVM特征...")
    train_texts = train_df['cleaned_review'].values
    train_labels = train_df['sentiment'].values
    
    token_to_idx, ratio = compute_ratio(train_texts, train_labels)
    X_train_nbsvm = create_nbsvm_features(train_texts, token_to_idx, ratio)
    X_test_nbsvm = create_nbsvm_features(test_df['cleaned_review'].values, token_to_idx, ratio)
    
    # 3.4 组合特征
    print("  - 组合特征...")
    X_train_combined = hstack([
        csr_matrix(X_train_nbsvm),
        X_train_tfidf,
        X_train_count
    ])
    X_test_combined = hstack([
        csr_matrix(X_test_nbsvm),
        X_test_tfidf,
        X_test_count
    ])
    
    y_train = train_df['sentiment'].values
    
    # 3.5 划分验证集
    print("\n步骤4: 划分验证集...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_combined, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 3.6 训练多个模型
    print("\n步骤5: 训练多个模型...")
    
    # 模型1: Logistic Regression with C=5.0
    print("  - 训练模型1: Logistic Regression (C=5.0)...")
    model1 = LogisticRegression(
        C=5.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
    model1.fit(X_train_combined, y_train)
    
    # 模型2: Logistic Regression with C=2.0
    print("  - 训练模型2: Logistic Regression (C=2.0)...")
    model2 = LogisticRegression(
        C=2.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
    model2.fit(X_train_combined, y_train)
    
    # 模型3: Logistic Regression with C=10.0
    print("  - 训练模型3: Logistic Regression (C=10.0)...")
    model3 = LogisticRegression(
        C=10.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
    )
    model3.fit(X_train_combined, y_train)
    
    # 3.7 集成预测
    print("\n步骤6: 集成预测...")
    
    # 在验证集上评估
    print("  - 在验证集上评估...")
    y_pred_proba1 = model1.predict_proba(X_val)[:, 1]
    y_pred_proba2 = model2.predict_proba(X_val)[:, 1]
    y_pred_proba3 = model3.predict_proba(X_val)[:, 1]
    
    # 归一化预测概率
    def normalize_proba(proba):
        return (proba - proba.mean()) / proba.std()
    
    y_pred_proba1_norm = normalize_proba(y_pred_proba1)
    y_pred_proba2_norm = normalize_proba(y_pred_proba2)
    y_pred_proba3_norm = normalize_proba(y_pred_proba3)
    
    # 简单平均集成
    y_pred_proba_ensemble = (y_pred_proba1_norm + y_pred_proba2_norm + y_pred_proba3_norm) / 3
    
    # 评估集成结果
    auc = roc_auc_score(y_val, y_pred_proba_ensemble)
    y_pred = (y_pred_proba_ensemble >= 0).astype(int)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"  集成模型验证集AUC: {auc:.4f}")
    print(f"  集成模型验证集准确率: {accuracy:.4f}")
    
    # 3.8 交叉验证
    print("\n步骤7: 交叉验证...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    accuracy_scores = []
    
    for train_idx, val_idx in skf.split(X_train_combined, y_train):
        X_train_cv = X_train_combined[train_idx]
        y_train_cv = y_train[train_idx]
        X_val_cv = X_train_combined[val_idx]
        y_val_cv = y_train[val_idx]
        
        # 训练三个模型
        m1 = LogisticRegression(C=5.0, solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)
        m2 = LogisticRegression(C=2.0, solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)
        m3 = LogisticRegression(C=10.0, solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)
        
        m1.fit(X_train_cv, y_train_cv)
        m2.fit(X_train_cv, y_train_cv)
        m3.fit(X_train_cv, y_train_cv)
        
        # 集成预测
        p1 = m1.predict_proba(X_val_cv)[:, 1]
        p2 = m2.predict_proba(X_val_cv)[:, 1]
        p3 = m3.predict_proba(X_val_cv)[:, 1]
        
        # 归一化
        p1_norm = normalize_proba(p1)
        p2_norm = normalize_proba(p2)
        p3_norm = normalize_proba(p3)
        
        p_ensemble = (p1_norm + p2_norm + p3_norm) / 3
        
        # 评估
        auc = roc_auc_score(y_val_cv, p_ensemble)
        acc = accuracy_score(y_val_cv, (p_ensemble >= 0).astype(int))
        
        auc_scores.append(auc)
        accuracy_scores.append(acc)
    
    print(f"5折交叉验证AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"5折交叉验证准确率: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    
    # 3.9 预测测试集
    print("\n步骤8: 预测测试集...")
    test_pred_proba1 = model1.predict_proba(X_test_combined)[:, 1]
    test_pred_proba2 = model2.predict_proba(X_test_combined)[:, 1]
    test_pred_proba3 = model3.predict_proba(X_test_combined)[:, 1]
    
    # 归一化
    test_pred_proba1_norm = normalize_proba(test_pred_proba1)
    test_pred_proba2_norm = normalize_proba(test_pred_proba2)
    test_pred_proba3_norm = normalize_proba(test_pred_proba3)
    
    # 集成预测
    test_pred_proba_ensemble = (test_pred_proba1_norm + test_pred_proba2_norm + test_pred_proba3_norm) / 3
    test_pred = (test_pred_proba_ensemble >= 0).astype(int)
    
    # 3.10 创建提交文件
    print("\n步骤9: 创建提交文件...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件已生成: submission.csv")
    print(f"提交文件大小: {len(submission)}")

if __name__ == "__main__":
    main()
