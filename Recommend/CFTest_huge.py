import numpy as np
import CF
import time
import random


def prepare_train_data(file_name):
    user_count = 0
    item_count = 0
    user_map = {}
    item_map = {}
    train_data = None
    try:
        with open(file_name, "r", encoding='UTF-8') as file:
            # 构造映射表
            for line in file:
                line_list = line.split()
                if line_list[0] not in user_map:
                    user_map[line_list[0]] = user_count
                    user_count += 1
                if line_list[1] not in item_map:
                    item_map[line_list[1]] = item_count
                    item_count += 1
            # 构造训练矩阵
            train_data = np.zeros((len(user_map), len(item_map)), dtype=int)
            file.seek(0)
            for line in file:
                line_list = line.split()
                train_data[user_map[line_list[0]], item_map[line_list[1]]] = int(line_list[2])
    except IOError as e:
        print(e)

    return user_map, item_map, train_data


start = time.time()
user_map, item_map, train_data = prepare_train_data("train_cut.dat")
end = time.time()
print("time of preparing data(s)", end - start)

# 砍小数据量,这里需要把不相关的物品也一起删除了。
user_count = len(user_map) // 10
train_data = train_data[:user_count, ]
item_index = np.count_nonzero(train_data, axis=0) > 0
train_data = train_data[:, item_index]

# 预先训练的版本
# User base CF
# start = time.time()
# cf_user_base = CF.CFUserBase(train_data)
# end = time.time()
# print("time of training UserBase CF model(s)", end - start)

# start = time.time()
# # print("predicting rate with UserBase CF model：", cf_user_base.predict(user_map['a'], item_map['1']))
# print("predicting rate with UserBase CF model：", cf_user_base.predict(user_map['R6vb0FtmClhfwajs_AuusQ'], item_map['tulUhFYMvBkYHsjmn30A9w']))
# end = time.time()
# print("time of predicting an item with UserBase CF model(s)", end - start)


# print("---------------")
# Item base CF
# start = time.time()
# cf_item_base = CF.CFItemBase(train_data)
# end = time.time()
# print("time of training ItemBase CF model(s)", end - start)

# start = time.time()
# print("predicting rate with ItemBase CF model：", cf_item_base.predict(4, 200))
# end = time.time()
# print("time of predicting an item with ItemBase CF model(s)", end - start)


# no training
error = 0
error2 = 0
test_times = 200
cf = CF.CFUserBase(train_data)
start = time.time()
for i in range(test_times):
    rand_user = random.randint(0, user_count - 1)
    j = 0
    while train_data[rand_user, j] == 0:
        j += 1
    to_predict = train_data[rand_user, j]
    train_data[rand_user, j] = 0
    print("-----user_id-", rand_user, ", item_id-", j, "-----")
    print("origin rate：", to_predict)
    my_predict = cf.predict(rand_user, j)
    print("predicting rate with ItemBase CF model：", my_predict)
    error += abs(my_predict - to_predict)
    error2 += abs(3 - to_predict)
    train_data[rand_user, j] = to_predict

end = time.time()
print("time cost of ", test_times, " times test ItemBase CF model(s)", end - start)
print("error is ", error / test_times)
print("error2 is ", error2 / test_times)

