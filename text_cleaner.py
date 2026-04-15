import re
import nltk
from nltk.corpus import stopwords

# 下载停用词
nltk.download('stopwords')

# 加载英文停用词
stop_words = set(stopwords.words('english'))

# 否定词列表
negation_words = {'not', 'no', 'never', 'nor', "n't", 'cannot', 'couldnt', 'shouldnt', 'wouldnt', 'wont', 'dont', 'doesnt', 'didnt'}

def remove_html_tags(text):
    """移除HTML标签"""
    return re.sub(r'<[^>]+>', '', text)

def remove_urls(text):
    """移除URL"""
    return re.sub(r'http\S+|www\S+', '', text)

def remove_emails(text):
    """移除邮箱"""
    return re.sub(r'\S+@\S+', '', text)

def remove_punctuation(text):
    """移除标点符号"""
    return re.sub(r'[\W_]', ' ', text)

def handle_negation(text):
    """处理否定词，将否定词和后续词标记为否定形式"""
    words = text.split()
    processed_words = []
    negation = False
    for word in words:
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

def remove_stopwords(text):
    """移除停用词"""
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def clean_text(text):
    """完整的文本清洗流程"""
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = text.lower()
    text = remove_punctuation(text)
    text = handle_negation(text)
    text = remove_stopwords(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == '__main__':
    test_text = "<br /><br />This movie is not bad at all! I really enjoyed it."
    print("原始文本:", test_text)
    print("清洗后:", clean_text(test_text))
