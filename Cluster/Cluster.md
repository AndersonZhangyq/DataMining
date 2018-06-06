# 短文本聚类

## 评价指标：NMI(Normalized Mutual Information)

指标效果：已知聚类标签与真实标签，互信息（mutual information）能够测度两种标签排列之间的相关性，同时忽略标签中的排列。

假设对于N个样本，存在两种标签分配策略，分别记做U，V。
定义U和V的信息熵为：
$$
H(U) = \sum_{i=1}^{|U|} P(i)\log{P(i)}
$$
其中，$P(i) = |U_i| / N$
$$
H(V) = \sum_{i=1}^{|V|} P'(i)\log{P'(i)}
$$
其中，$P(i) = |V_i| / N$

则互信息MI定义为：
$$
MI(U,V)=\sum_{i = 1}^{|U|}\sum_{j = 1}^{|V|} P(i,j)\log{\frac{P(i,j)}{P'(i)P'(j)}}
$$
其中，$P(i,j) = \frac{|U_i \cap V_i|}{N}$

对MI进行标准化后得到NMI:
$$
NMI(U,V) = \frac{MI(U, V)}{\sqrt{H(U)H(V)}}
$$

## Baseline: 随机分类

一共65个不同的类别，读入所有数据，随机分类，进行10次实验，取平均值：

```python
### Core Code

def execute_cluster(tokens_list):

    #... other codes ...

    clusters_list = deepcopy(tokens_list)
    for d in clusters_list:
        # Random int taken from [0,65)
        d['cluster'] = np.random.randint(0, 65)

    return clusters_list

def evaluate_train_result():
    train_tokens_list = load_array(train_tokens_file)
    train_clusters_list = execute_cluster(train_tokens_list)
    train_topics_list = load_array(train_topics_file)

    #... add return ...
    return calculate_nmi(train_topics_list, train_clusters_list)

sum = 0.0
for _ in range(10):
    sum += evaluate_train_result()
print("Mean NMI: {}".format(sum / 10))
```

输出：

```
nmi:0.1004
nmi:0.0977
nmi:0.1006
nmi:0.0989
nmi:0.0998
nmi:0.0997
nmi:0.1008
nmi:0.1041
nmi:0.0998
nmi:0.1002
Mean NMI: 0.10019629653751874
```

## TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) 是一种文本挖掘技术，用于对文本进行分类。TF-IDF 将文本编码为带有权重的词向量。
$$
TF(i,j) = \frac{freq(i,j)}{termPerDoc(j)}
$$

$freq(i,j)$: 词 $i$ 在文档 $j$ 中出现的次数

$termPerDoc(j)$: 文档 j 中 term 的个数

$$
IDF(i) = \ln{\frac{N}{n_i}}
$$


$N$: 文档的数目

$n_i$: 出现词 $i$ 的文档的数量

```python
class TF_IDF:
    def __init__(self, token_list):
        self.token_list = token_list  # type: list(dict)
        self.IDF = self.__calculate_idf()
        self.TF = self.__calculate_tf()

    # Calculate the IDF for each term
    def __calculate_idf(self):
        num_of_doc = len(self.token_list)
        # get all unique terms
        terms = [self.token_list[t]['tokenids'] for t in range(len(self.token_list))]
        import itertools
        # 1. concat all term-list in each document
        # 2. make term unique with set()
        # 3. turn set to list
        terms = list(set(list(itertools.chain(*terms))))

        # get dictionary as {term: doc_cnt}
        term_doc = {t: 0 for t in terms}
        for row in self.token_list:
            for token in row['tokenids']:
                term_doc[token] += 1
        import math

        # get IDF matrix as {term: IDF}
        return {k: math.log(num_of_doc / v) for k, v in term_doc.items()}

    # Calculate TF for each term in each document
    def __calculate_tf(self):
        # get number of terms in each document
        term_per_doc = {t['docid']: len(t['tokenids']) for t in self.token_list}

        # start calculate TF
        TF = {}
        for row in self.token_list:  # type: dict
            cur_doc_id = row['docid']
            TF[cur_doc_id] = {}
            cur_terms = row['tokenids']  # type: list
            # 1. get number of terms in current document
            term_in_cur_doc = term_per_doc[cur_doc_id]
            for term in cur_terms:  # type: int
                # 2. iterate through all terms in current document, calculate TF
                TF[cur_doc_id][term] = cur_terms.count(term) / term_in_cur_doc
        return TF

    # Calculate TF-IDF for each term in document
    def get_result(self):
        # result is a list of dict of type {docid: {}}
        result = {}
        docid = [t['docid'] for t in self.token_list]
        for id in docid:
            # tmp is a dict of type {term: TF-IDF}
            tmp = {}
            for term, tf in self.TF[id].items():
                tmp[term] = tf * self.IDF[term]
            result[id] = tmp
        return result

```

训练集上NMI:
```
nmi:0.7543
```


## LDA

抛开数学推导部分，LDA还是挺容易理解的。

主题模型用于从一系列文档中发现抽象主题的一类数学模型。该模型使用的思路比较新奇，通过生成文档的方式来建模。通过选择文档中的主题，并选择该主题所对应的词的方式来生成文档。然而如何选择主题，如何选择词，这些选择方式的复杂程度区分了不同的模型。

首先从几个简单的模型开始

#### 一元模型(Unigram model)

对于文档 $W = (w_1, w_1, w_3, ..., w_n)$，生成该文档的概率是
$$
P(W) = \prod_{n = 1}^{N} P(w_n)
$$

#### 混合一元模型(Mixture of unigrams model)

给某个文档先选择一个主题z，再根据该主题生成文档，该文档中的所有词都来自一个主题。生成文档的概率为：
$$
P(W) = \sum_{z}P(z) \prod_{n = 1}^{N}P(w_n|z)
$$

#### pLSA模型

在混合一元模型中，假定一篇文档只由一个主题生成，可实际中，一篇文章往往有多个主题，只是这多个主题各自在文档中出现的概率大小不一样。在pLSA中，假设文档由多个主题生成。

假设现在有三个主题，其选中的概率分别是{A: 0.3, B: 0.2, C: 0.5}，又假设主题A下有词，每个词选中的概率分别是{W_A: 0.4, W_B: 0.5, W_C: 0.1}，则首先选择某一主题，假设A，其次选择该主题A下的词。在这里无论是主题的分布还是词的分布都是固定的。

#### LDA模型

在pLSA模型中，对主题和词的分布都是固定的，而在LDA中，即使这个分布都是不固定的，是从先验信息狄利克雷分布中抽出来的。

主要代码：
``` python
class LDA:
    def __init__(self, doc_term_matrix):
        self.doc_term_matrix = doc_term_matrix
        import lda
        self.model = lda.LDA(n_topics=65, n_iter=2600)


    def getResult(self):
        result = self.model.fit(self.doc_term_matrix)
        # topic_word = self.model.topic_word_  # model.components_ also works
        # n_top_words = 8
        # for i, topic_dist in enumerate(topic_word):
        #     print('Topic {}: {}'.format(i, ' '.join(topic_dist)))
        import matplotlib.pyplot as plt
        plt.plot(self.model.loglikelihoods_[5:])
        plt.show()
        doc_topic = self.model.doc_topic_
        result = []
        for i in doc_topic:
            result.append(i.argmax())
        return result

```

训练集上NMI:
```
nmi:0.8120
```