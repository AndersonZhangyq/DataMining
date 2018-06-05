import itertools


class TextCluster:
    def __init__(self, token_list):
        self.token_list = token_list  # type: list(dict)
        self.doc_list = [t['docid'] for t in token_list]
        self.term_list = list(
            set(list(itertools.chain(*[self.token_list[t]['tokenids'] for t in range(len(self.token_list))]))))
        self.term_index = dict(zip(self.term_list, list(range(0, len(self.token_list)))))
        self.doc_index = dict(zip(self.doc_list, list(range(0, len(self.doc_list)))))

    def useTFIDF(self):
        from Cluster import TF_IDF
        tf_idf = TF_IDF.TF_IDF(self.token_list)
        result = tf_idf.get_result()
        import numpy as np
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
