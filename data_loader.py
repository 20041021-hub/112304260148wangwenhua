import pandas as pd

def load_train_data():
    """加载训练集数据"""
    train_df = pd.read_csv('labeledTrainData.tsv/labeledTrainData.tsv', sep='\t')
    print(f"训练集加载完成，共{len(train_df)}条数据")
    return train_df

def load_test_data():
    """加载测试集数据"""
    test_df = pd.read_csv('testData.tsv/testData.tsv', sep='\t')
    print(f"测试集加载完成，共{len(test_df)}条数据")
    return test_df
