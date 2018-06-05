import numpy as np
import math


class CF:
    min_sim = 0.5

    def __init__(self, data):
        assert isinstance(data, np.ndarray)
        self._data = data
        n = data.shape[0]  # 行数
        self._is_trained = np.array([False] * n)
        self._sim_arr = np.zeros((n, n))
        self._mean_arr = np.zeros((n, n))
        # 统计每行的均值
        self._mean_arr_gob = np.sum(data, axis=1) / np.count_nonzero(data, axis=1)

    def _sim(self, row_a, row_b):
        return 0.0

    @staticmethod
    def get_mean(data, row_a, row_b):
        index = ((data[row_a, ] != 0) & (data[row_b, ] != 0))
        if np.sum(index) == 0:
            return 0, 0
        a = data[row_a, index]
        b = data[row_b, index]
        return np.mean(a), np.mean(b)

    def train(self):
        # 计算每行相对的局部均值
        n = self._data.shape[0]  # 行数
        for i in range(n):
            for j in range(i + 1, n):
                self._mean_arr[i, j], self._mean_arr[j, i] = CF.get_mean(self._data, i, j)
                
        # 计算每行的相似度
        self._sim_arr = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                tmp = self._sim(i, j)
                # 保证矩阵中的相似度都高于最小要求
                if tmp < CF.min_sim:
                    tmp = 0
                self._sim_arr[i, j] = tmp

        # 所有行标记为已训练
        self._is_trained = np.array([True] * n)

    def predict(self, row_a, col_p):
        n = self._data.shape[0]  # 行数
        # 假如该行未训练过
        if not self._is_trained[row_a]:
            # 计算该行局部均值
            for j in range(n):
                mean_a, mean_b = CF.get_mean(self._data, row_a, j)
                self._mean_arr[row_a, j] = mean_a
                self._mean_arr[j, row_a] = mean_b
            self._mean_arr[row_a, row_a] = np.mean(self._mean_arr[self._mean_arr > 0])

            # 计算该行相似度
            sim_arr = self._sim_arr
            for j in range(n):
                tmp = self._sim(row_a, j)
                if tmp < CF.min_sim:
                    tmp = 0
                sim_arr[row_a, j] = tmp
            # 标记为已训练
            self._is_trained[row_a] = True

        user_index = self._data[:, col_p] > 0
        numerator = np.sum(self._sim_arr[row_a, user_index].reshape(1, -1) *
                           (self._data[user_index, col_p].reshape(1, -1) - self._mean_arr[user_index, row_a].reshape(1, -1)))
        denominator = np.sum(self._sim_arr[row_a, user_index])
        if denominator == 0:
            return self._mean_arr[row_a, row_a]
        result = self._mean_arr[row_a, row_a] + numerator / denominator
        # print("offset to mean is ", numerator / denominator)
        return result


class CFUserBase(CF):
    def __init__(self, data):
        CF.__init__(self, data)

    def _sim(self, row_a, row_b):
        return CFUserBase.pearson(self._data, row_a, row_b)

    @staticmethod
    def pearson(data, row_a, row_b):
        index = ((data[row_a, ] != 0) & (data[row_b, ] != 0))
        if np.sum(index) == 0:
            return 0
        a = data[row_a, index]
        b = data[row_b, index]
        # a_offset = a - self._mean_arr[row_a]
        # b_offset = b - self._mean_arr[row_b]
        a_offset = a - np.mean(a)
        b_offset = b - np.mean(b)
        numerator = np.sum(a_offset * b_offset)
        denominator = math.sqrt(np.sum(a_offset ** 2) * np.sum(b_offset ** 2))
        if denominator == 0:
            a_offset += 0.01
            b_offset += 0.01
            numerator = np.sum(a_offset * b_offset)
            denominator = math.sqrt(np.sum(a_offset ** 2) * np.sum(b_offset ** 2))
        result = numerator / denominator  # 也许最小相似度限制可以放这里过滤？
        return result


class CFItemBase(CF):
    def __init__(self, data):
        CF.__init__(self, data.T)

    def predict(self, row_a, col_p):
        return CF.predict(self, col_p, row_a)

    def _sim(self, row_a, row_b):
        return CFItemBase.cosine(self._data, row_a, row_b)

    @staticmethod
    def cosine(data, row_a, row_b):
        a = data[row_a, ]
        b = data[row_b, ]
        numerator = np.sum(a * b)
        denominator = math.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        result = numerator / denominator
        return result


# 采用全局均值
def predict_without_training(data, row_a, col_p):
    assert isinstance(data, np.ndarray)
    n = data.shape[0]  # 行数
    mean_arr = np.sum(data, axis=1) / np.count_nonzero(data, axis=1)
    sim_arr = np.zeros((1, n))
    for i in range(n):
        tmp = CFUserBase.pearson(data, row_a, i)
        if tmp < CF.min_sim:
            tmp = 0
        sim_arr[0, i] = tmp

    user_index = data[:, col_p] > 0
    numerator = np.sum(sim_arr[0, user_index] *
                       (data[user_index, col_p].reshape(1, -1) - mean_arr[user_index].reshape(1, -1)))
    denominator = np.sum(sim_arr)
    result = mean_arr[row_a] + numerator / denominator
    return result


# try:
#     with open("train.dat", "r", encoding='UTF-8') as file1, open("train_cut.dat", "w", encoding="UTF-8") as file2:
#         for line in file1:
#             line_list = line.split()
#             file2.write(line_list[0] + ' ' + line_list[1] + ' ' + line_list[2] + ' \n')
# except IOError as e:
#     print(e)