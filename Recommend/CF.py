import pandas as pd
import numpy as np
import DataUtility


class collaborativeFiltering:

    def __init__(self, trainData, limit):
        self.userList = list(trainData.index)
        self.itemList = list(trainData.columns)
        self.userSize = len(self.userList)
        self.itemSize = len(self.itemList)
        self.user_item_matrix = trainData.as_matrix()
        self.limit = limit

    def predict(self, testData):
        predicted = {}
        # 允许出现多组数据，对每组数据分别做预测
        outPut = open("predicted.out", mode="w")
        for t in testData:
            testUser = self.userList.index(t[0].strip())
            testItem = self.itemList.index(t[1].strip())
            predicted_v = self.predictCore(testUser, testItem)
            predicted_v = max(int(predicted_v + 0.5), 0)
            # print("{} Real: {} Predicted: {}".format(str(t), t[2], predicted_v))
            outPut.write("{} {} {}\n".format(t[0].strip(), t[1].strip(), predicted_v))
            print("{} {} {}".format(t[0].strip(), t[1].strip(), predicted_v))
            predicted[str(t)] = predicted_v
        outPut.close()
        return predicted

    def predictCore(self, testUser, testItem):
        # 评分预测值计算式中的分子部分
        predict_up = 0
        # 评分预测值计算式中的分母部分
        predict_down = 0
        user_a = self.user_item_matrix[testUser]
        for user in range(self.userSize):
            if user == testUser:
                continue
            else:
                user_b = self.user_item_matrix[user]
                # 如果用户 b 未对 testItem 做出评价，那么不考虑用户 b
                if user_b[testItem] == 0:
                    continue
                item_both_rated = [i for i in range(self.itemSize) if user_a[i] != 0 and user_b[i] != 0]
                # 如果用户 a 和 用户 b 没有评价重叠的商品，那么不考虑用户 b
                if len(item_both_rated) == 0:
                    continue
                # 获取用户 a 和用户 b 都评价的商品的评价值
                user_a_ = [user_a[i] for i in item_both_rated]
                user_b_ = [user_b[i] for i in item_both_rated]
                # 根据用户 a 和用户 b 都评价的商品的评价值，计算Pearson 相关系数
                sim = np.corrcoef(user_a_, user_b_)[0, 1]
                # 如果相关系数大于阈值，那么加入到评分预测值计算中
                if sim >= self.limit:
                    predict_up += sim * (user_b[testItem] - np.mean(user_b_))
                    predict_down += sim
        return 1 if predict_down == 0 else predict_up / predict_down + np.sum(user_a) / np.count_nonzero(user_a)


df_train = DataUtility.getTrainData()
# df_train = df_train.iloc[0:int(df_train.shape[0] / 10), 0:int(df_train.shape[1])]
# df_test = DataUtility.getTestData()
CF = collaborativeFiltering(df_train, 0.5)
test_data = []
user = list(df_train.index)
item = list(df_train.columns)
for i in range(int(df_train.shape[0] / 10)):
    for j in range(df_train.shape[1]):
        if df_train.iloc[i, j] == 0:
            continue
        else:
            test_data.append([user[i], item[j], df_train.iloc[i, j]])
            break
test_data = [[t[0], t[1], t[2]] for t in test_data if t[2] != 0]
p = CF.predict(test_data)
# p = CF.predict(df_test.as_matrix())
print(p)
