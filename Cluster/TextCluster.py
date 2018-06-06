import itertools

import numpy as np


class TextCluster:
    def __init__(self, token_list):
        self.token_list = token_list  # type: list(dict)
        self.doc_list = [t['docid'] for t in token_list]
        self.term_list = list(
            set(list(itertools.chain(*[self.token_list[t]['tokenids'] for t in range(len(self.token_list))]))))
        self.term_index = dict(zip(self.term_list, list(range(0, len(self.token_list)))))
        self.doc_index = dict(zip(self.doc_list, list(range(0, len(self.doc_list)))))

    def useTFIDF(self):
        import TF_IDF
        tf_idf = TF_IDF.TF_IDF(self.token_list)
        result = tf_idf.get_result()
        data = np.zeros((len(self.doc_list), len(self.term_list)))
        for doc_id in result.keys():
            cur_tf_idf = result[doc_id]
            doc_i = self.doc_index[doc_id]
            for term_id in cur_tf_idf.keys():
                data[doc_i, self.term_index[term_id]] = cur_tf_idf[term_id]
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=65)
        data = kmeans.fit_predict(data)
        clustered = []
        for doc_id in result.keys():
            tmp = {"docid": doc_id, "cluster": data[self.doc_index[doc_id]]}
            clustered.append(tmp)
        return clustered

    def useLDA(self):
        import LDA
        doc_term_matrix = np.zeros((len(self.doc_list), len(self.term_list)), dtype=np.int8)
        for record in self.token_list:  # type: dict
            doc_i = self.doc_index[record['docid']]
            terms = record['tokenids']  # type: list
            for term in terms:
                doc_term_matrix[doc_i, self.term_index[term]] = terms.count(term)
        lda = LDA.LDA(doc_term_matrix)
        result = lda.getResult()
        clustered = []
        inverted_doc_index = {v:k for k, v in self.doc_index.items()}
        for i in range(len(self.doc_list)):
            tmp = {"docid": inverted_doc_index[i], "cluster": result[i]}
            clustered.append(tmp)
        return clustered