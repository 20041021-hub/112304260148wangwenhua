import re
import nltk
from nltk.corpus import stopwords

# 下载停用词
nltk.download('stopwords')

# 加载英文停用词
stop_words = set(stopwords.words('english'))

# 否定词列表
negation_words = {'not', 'no', 'never', 'nor', "n't", 'cannot'}

# 情感词增强
emotion_markers = {'!', '?', '!!!', '???'}

# 移除HTML标签
def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# 小写化
def to_lowercase(text):
    return text.lower()

# 标点处理
def handle_punctuation(text):
    # 保留情感相关的标点，如感叹号和问号
    text = re.sub(r'[.,;:]', ' ', text)
    # 保留多个感叹号或问号作为情感标记
    text = re.sub(r'(!+)', r' \1 ', text)
    text = re.sub(r'(\?+)', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 处理否定词
# 处理否定词，将否定词与后续词合并，增强否定效果
def handle_negation(text):
    words = text.split()
    processed_words = []
    negation = False
    for word in words:
        if word in negation_words:
            negation = True
            processed_words.append(word)
        else:
            if negation:
                processed_words.append('not_' + word)
                negation = False
            else:
                processed_words.append(word)
    return ' '.join(processed_words)

# 停用词处理（保留否定词和情感标记）
def remove_stopwords(text):
    words = text.split()
    filtered_words = []
    for word in words:
        # 保留否定词、情感标记和以not_开头的词
        if word not in stop_words or word in negation_words or word in emotion_markers or word.startswith('not_'):
            filtered_words.append(word)
    return ' '.join(filtered_words)

# 完整的预处理流程
def preprocess_text(text):
    text = remove_html_tags(text)
    text = to_lowercase(text)
    text = handle_punctuation(text)
    text = handle_negation(text)
    text = remove_stopwords(text)
    return text

# 测试预处理函数
if __name__ == '__main__':
    test_text = "<br /><br />This movie is not bad at all! I really enjoyed it."
    print("原始文本:", test_text)
    print("预处理后:", preprocess_text(test_text))
