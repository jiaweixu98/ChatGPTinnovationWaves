ChatGPT中文献中的关键词演变。
用什么方法抽关键词，我们关心什么样的关键词？
第一个图是看，大家是不是在做evaluation或者别的，还是开发新的算法，还是应用？
看它们随时间的变化！


chatGPT
图1：共现的词汇；context。
图2：model 出现次数最多的；
图3：GPT spread, 用LDA
Team affiliation.
Team industrial, academic.
core innovation, chatGPT. First Wave.

Paper. Keep on updating.

1960 diffusion of innovation. 对第一个wave的影响
ChatGPT.
ChatGPT, PaMl, 这些词有什么关系？
Team Size
Core
Context.
Competitor.
Content, dynamic,

做topic的这些人，他们之间会有竞争关系吗？竞争性很强，不引用有竞争关系的论文？数量上，就是
self-correction in science. The pace of AI. 发不出去。
其他的领域，从工具性的角度去看。使用了这些方法或技术。提升论文的 novelty. Apply. Open Alex.

1. Core ChatGPT content and dynamic topic analysis, keywords, and algorithms.
joint attention. keyword, topic modeling. Jian Xu.

2. when competitors (palm,) before and after, impact on chatGPT innovation topic changes.
出现前后；after, before, citation的变化。

3. after and before competitors appear, OpenAlex

4. Citing Topic dynamics.

Yan Zhang. Competitor.

第二个wave的开始，AI innovation 和 competitor的出现是有关系的。 chain of thought. tree of thought.
open source, closed source.

community, industrial.

core

Xu, J., Bu, Y., Ding, Y., Yang, S., Zhang, H., Yu, C., & Sun, L. (2018). Understanding the formation of interdisciplinary research from the perspective of keyword evolution: a case study on joint attention. Scientometrics, 117(2), 973-995.


关键词抽取：
keyword evolution；



术语抽取；
@inproceedings{ushio-etal-2021-kex,
    title={{B}ack to the {B}asics: {A} {Q}uantitative {A}nalysis of {S}tatistical and {G}raph-{B}ased {T}erm {W}eighting {S}chemes for {K}eyword {E}xtraction},
    author={Ushio, Asahi and Liberatore, Federico and Camacho-Collados, Jose},
        booktitle={Proceedings of the {EMNLP} 2021 Main Conference},
    year = {2021},
    publisher={Association for Computational Linguistics}
}


抽取的流程：
1. The documents are first tokenized into words by segtok
2. each word is stemmed to reduce it to its base form for comparison purpose by Porter Stemmer from NLTK
3. (ADJECTIVE)*(NOUN)+
4. stopword list taken from the official YAKE implementation
5. prior statistics including term frequency (tf)


词频，先看总体。

词频
看每个月，关键词的累积数量。（按月度算）。每篇文章的phrases

每篇文章抽5个关键词；
每个月总关键词数是5*当月文章数
计算每个月的top关键词数量

看abstract, 看 title