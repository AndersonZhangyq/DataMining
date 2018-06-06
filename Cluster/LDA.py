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
