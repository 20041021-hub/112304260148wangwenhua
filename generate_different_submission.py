import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    # 1. 加载数据
    print("加载数据...")
    train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
    test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
    
    # 2. 清洗文本（使用不同的清洗方法）
    print("清洗文本...")
    import re
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    def different_clean(text):
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 小写化
        text = text.lower()
        # 移除标点符号
        text = re.sub(r'[\W_]', ' ', text)
        # 移除停用词
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        # 保留否定词
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
    
    train_df['cleaned_review'] = train_df['review'].apply(different_clean)
    test_df['cleaned_review'] = test_df['review'].apply(different_clean)
    
    # 3. 特征工程（使用不同的参数）
    print("特征工程...")
    tfidf = TfidfVectorizer(
        max_features=40000,  # 不同的max_features
        ngram_range=(1, 3),   # 不同的ngram_range
        sublinear_tf=True,
        min_df=3,             # 不同的min_df
        max_df=0.7            # 不同的max_df
    )
    X_train = tfidf.fit_transform(train_df['cleaned_review'])
    X_test = tfidf.transform(test_df['cleaned_review'])
    y_train = train_df['sentiment'].values
    
    # 4. 训练模型（使用不同的参数）
    print("训练模型...")
    model = LogisticRegression(
        C=10.0,                # 不同的C值
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=3000          # 不同的max_iter
    )
    model.fit(X_train, y_train)
    
    # 5. 预测测试集
    print("预测测试集...")
    test_predictions = model.predict(X_test)
    
    # 6. 创建提交文件
    print("创建提交文件...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'sentiment': test_predictions
    })
    
    # 明确覆盖文件
    submission.to_csv('submission.csv', index=False)
    print(f"提交文件已覆盖: submission.csv")
    print(f"提交文件大小: {len(submission)}")
    
    # 验证文件内容
    print("验证文件内容...")
    sample = pd.read_csv('submission.csv').head()
    print("文件前5行:")
    print(sample)
    
    # 计算预测分布
    print("预测分布:")
    print(submission['sentiment'].value_counts())

if __name__ == "__main__":
    main()
