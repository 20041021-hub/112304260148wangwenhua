from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def create_count_vectorizer(max_features=20000, ngram_range=(1, 3)):
    """创建CountVectorizer"""
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=False
    )

def create_tfidf_vectorizer(max_features=50000, ngram_range=(1, 4), sublinear_tf=True, min_df=5, max_df=0.8):
    """创建TF-IDF Vectorizer"""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
        lowercase=False
    )

def fit_transform_vectorizer(vectorizer, train_texts):
    """拟合并转换文本"""
    return vectorizer.fit_transform(train_texts)

def transform_vectorizer(vectorizer, texts):
    """转换文本"""
    return vectorizer.transform(texts)
