import numpy as np


class LFM:

    def __init__(self, trainData, F, alpha=0.1, lmbd=0.1, max_iter=500):
        self.userList = list(trainData.index)
        self.itemList = list(trainData.columns)
        self.F = F
        self.P = []
        self.Q = []
        self.bu = []
        self.bi = []
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.trainData = trainData.as_matrix()
        self.init()

    def init(self):
        self.P = np.random.rand(len(self.userList), self.F)
        self.Q = np.random.rand(len(self.itemList), self.F)
        for i in range(len(self.userList)):
            self.bu.append(np.sum(self.trainData[i]) / np.count_nonzero(self.trainData[i]))
        for j in range(len(self.itemList)):
            self.bi.append(np.sum(self.trainData[:, j] / np.count_nonzero(self.trainData[:, j])))

    def train(self):
        for _ in range(self.max_iter):
            for user in range(len(self.userList)):
                user_rates = self.trainData[user]
                for item in range(len(user_rates)):
                    if user_rates[item] == 0:
                        continue
                    _rui = np.sum([self.P[user][f] * self.Q[item][f] for f in range(self.F)]) + self.bu[user] + self.bi[item]
                    err_ = user_rates[item] - _rui
                    self.bu[user] += self.alpha * (err_ - self.lmbd * self.bu[user])
                    self.bi[item] += self.alpha * (err_ - self.lmbd * self.bi[item])
                    for f in range(self.F):
                        self.P[user][f] += self.alpha * \
                                           (err_ * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * \
                                           (err_ * self.P[user][f] - self.lmbd * self.Q[item][f])
            self.alpha *= 0.9
