class LDA:
    def __init__(self, doc_term_matrix):
        self.doc_term_matrix = doc_term_matrix
        import lda
        # initialize LDA model
        self.model = lda.LDA(n_topics=65, n_iter=2600)


    def getResult(self):
        # train LDA
        result = self.model.fit(self.doc_term_matrix)
        import matplotlib.pyplot as plt
        plt.plot(self.model.loglikelihoods_[5:])
        plt.show()
        doc_topic = self.model.doc_topic_
        result = []
        # get topic for each document
        for i in doc_topic:
            # choose the index of largest possibility as the topic
            result.append(i.argmax())
        return result
