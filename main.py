from data_loader import load_train_data, load_test_data
from text_cleaner import clean_text
from feature_engineering import create_tfidf_vectorizer, fit_transform_vectorizer, transform_vectorizer
from model_trainer import create_logistic_regression, train_model, cross_validate_model, evaluate_model
from predictor import predict_test, create_submission
from sklearn.model_selection import train_test_split

def main():
    # 1. 加载数据
    print("步骤1: 加载数据...")
    train_df = load_train_data()
    test_df = load_test_data()
    
    # 2. 清洗文本
    print("\n步骤2: 清洗文本...")
    train_df['cleaned_review'] = train_df['review'].apply(clean_text)
    test_df['cleaned_review'] = test_df['review'].apply(clean_text)
    
    # 3. 特征工程
    print("\n步骤3: 特征工程...")
    vectorizer = create_tfidf_vectorizer(
        max_features=50000,
        ngram_range=(1, 4),
        sublinear_tf=True,
        min_df=5,
        max_df=0.8
    )
    
    # 转换训练集
    X_train = fit_transform_vectorizer(vectorizer, train_df['cleaned_review'])
    y_train = train_df['sentiment'].values
    
    # 转换测试集
    X_test = transform_vectorizer(vectorizer, test_df['cleaned_review'])
    
    # 4. 划分验证集
    print("\n步骤4: 划分验证集...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 5. 创建并训练模型
    print("\n步骤5: 训练模型...")
    model = create_logistic_regression(
        C=5.0,
        solver='liblinear',
        class_weight='balanced'
    )
    
    # 6. 交叉验证
    print("\n步骤6: 交叉验证...")
    cross_validate_model(model, X_train, y_train, n_splits=3)
    
    # 7. 训练模型
    model = train_model(model, X_train, y_train)
    
    # 8. 评估模型
    print("\n步骤7: 评估模型...")
    evaluate_model(model, X_val, y_val)
    
    # 9. 预测测试集
    print("\n步骤8: 预测测试集...")
    test_predictions = predict_test(model, X_test)
    
    # 10. 创建提交文件
    print("\n步骤9: 创建提交文件...")
    create_submission(test_df['id'], test_predictions, 'submission.csv')

if __name__ == "__main__":
    main()
