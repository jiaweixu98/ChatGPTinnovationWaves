import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
import nltk

# 读取CSV文件
df = pd.read_csv('C:/Users/ytl/Desktop/LLM.csv')

# 文本预处理函数
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())  # 分词并转小写
    stop_words = set(stopwords.words('english'))  # 英文停用词
    tokens = [w for w in tokens if not w in stop_words and w.isalpha()]  # 移除停用词和非字母字符
    return ' '.join(tokens)

# 应用预处理
df['processed_title'] = df['title'].apply(preprocess_text)
df['processed_abstract'] = df['abstract'].apply(preprocess_text)

# 使用Sentence Transformer库提供的预训练模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 对标题和摘要进行编码，生成句子的向量表示
sentences = df['title'] + " " + df['abstract']
sentence_embeddings = model.encode(sentences)

# 确定最佳聚类数量
range_n_clusters = range(5, 20)
silhouette_scores = []
for n_clusters in range_n_clusters:
    clustering_model = KMeans(n_clusters=n_clusters)
    clustering_model.fit(sentence_embeddings)
    score = silhouette_score(sentence_embeddings, clustering_model.labels_)
    silhouette_scores.append(score)

# 选择具有最高轮廓系数的聚类数
best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
print(f"最佳聚类数量为: {best_n_clusters}")

# 使用最佳聚类数进行聚类
clustering_model = KMeans(n_clusters=best_n_clusters)
cluster_labels = clustering_model.fit_predict(sentence_embeddings)

# 将聚类标签添加到原始数据框中
df['cluster'] = cluster_labels

# 为了可视化，我们可以将聚类结果降维到2维
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(sentence_embeddings)

# 绘制降维后的数据点，以颜色表示不同的聚类
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='viridis', marker='o')
plt.colorbar()
plt.title('Clusters of documents')
plt.xlabel('PCA feature 1')
plt.ylabel('PCA feature 2')
plt.show()

# 保存聚类结果到新的CSV文件
df.to_csv('C:/Users/ytl/Desktop/clustered_papers.csv', index=False)