import pandas as pd
from gensim.models import Word2Vec
from collections import Counter
import itertools

# 读取数据
data = pd.read_csv('C:/Users/ytl/Desktop/clustered_papers.csv')

# 准备数据：将关键词字符串分割成列表
data['keywords'] = data['keywords'].apply(lambda x: x.split(','))

# 创建Word2Vec模型来学习关键词的语义关系
model = Word2Vec(sentences=data['keywords'], vector_size=100, window=5, min_count=1, workers=4)

# 全局变量来存储已经选择的主题词组
global_selected_phrases = []

# 定义一个函数来计算两个词组的相似度
def similarity(word_group1, word_group2):
    words1 = word_group1.split()
    words2 = word_group2.split()
    sim = model.wv.n_similarity(words1, words2)
    return sim

# 定义一个函数来过滤相似的词组
def filter_similar_phrases(phrases, threshold=0.8):
    filtered_phrases = []
    for phrase in phrases:
        if all(similarity(phrase, fp) < threshold for fp in global_selected_phrases):
            filtered_phrases.append(phrase)
            global_selected_phrases.append(phrase)  # 更新全局列表
    return filtered_phrases

# 定义一个函数来找到每个聚类的主题
def extract_themes(cluster_keywords):
    all_keywords = list(itertools.chain(*cluster_keywords))
    # 使用Counter来找到最常见的关键词组
    most_common_phrases = [phrase for phrase, count in Counter(all_keywords).most_common(15)]
    # 过滤相似的词组
    filtered_phrases = filter_similar_phrases(most_common_phrases)
    return filtered_phrases

# 应用到每个聚类，并转换结果为字符串格式以便保存
themes = data.groupby('cluster')['keywords'].apply(extract_themes).reset_index()
themes['themes'] = themes['keywords'].apply(lambda x: ', '.join(x))

# 保存到CSV文件
themes.to_csv('C:/Users/ytl/Desktop/cluster_themes.csv', index=False)
