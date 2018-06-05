# 短文本聚类
评价指标：NMI(Normalized Mutual Information)

```@todo``` Need more detail here

Baseline: 随机分类

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

    def get_result(self):
        result = {}
        docid = [t['docid'] for t in self.token_list]
        for id in docid:
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