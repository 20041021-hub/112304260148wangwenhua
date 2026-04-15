import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    # 1. 加载数据
    print("加载数据...")
    train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
    test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
    
    # 2. 清洗文本
    print("清洗文本...")
    from text_cleaner import clean_text
    train_df['cleaned_review'] = train_df['review'].apply(clean_text)
    test_df['cleaned_review'] = test_df['review'].apply(clean_text)
    
    # 3. 特征工程
    print("特征工程...")
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
    
    # 4. 训练模型
    print("训练模型...")
    model = LogisticRegression(
        C=5.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=2000
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

if __name__ == "__main__":
    main()
