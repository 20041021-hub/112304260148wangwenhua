import pandas as pd
import numpy as np
import joblib

# 加载测试数据
test_df = pd.read_csv('processed_test.csv')
test_vectors = np.load('test_vectors.npy')

# 加载最佳模型
model = joblib.load('best_model.joblib')

# 预测测试集
print("预测测试集...")
y_pred = model.predict(test_vectors)

# 创建提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': y_pred
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)

print("提交文件已生成: submission.csv")
print(f"提交文件大小: {len(submission)}")
