class TF_IDF:
    def __init__(self, token_list):
        self.token_list = token_list  # type: list(dict)
        self.IDF = self.__calculate_idf()
        self.TF = self.__calculate_tf()
        # self.print_result()

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

    # def print_result(self):
    #     file = open('./result.out', 'w')
    #     for docid in self.result.keys():
    #         file.write("{} {}".format(docid, str(self.result[docid])))
    #         file.write("\n")
    #     file.close()
