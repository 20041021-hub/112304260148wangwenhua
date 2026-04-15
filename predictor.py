import pandas as pd

def predict_test(model, X_test):
    """预测测试集"""
    return model.predict(X_test)

def create_submission(test_ids, predictions, filename='submission.csv'):
    """创建提交文件"""
    submission = pd.DataFrame({
        'id': test_ids,
        'sentiment': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"提交文件已生成: {filename}")
    print(f"提交文件大小: {len(submission)}")
    return submission
