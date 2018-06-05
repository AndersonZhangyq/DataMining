import numpy, pandas, time


def getTrainData():
    print("-------- Prepare train data --------")
    startT = time.time()
    train_data = open('train_data_for_test.dat', encoding='utf-8', mode='r')
    train_data = [l.split(' ') for l in train_data.readlines()]
    train_data = [{'userId': t[0], 'itemId': t[1], 'score': t[2], 'word': t[3], 'each': t[4:-1]}
                  for t in train_data]

    # 利用 DataFrame 构造 用户-物品 评分矩阵
    user_list = list(set([t['userId'] for t in train_data]))
    item_list = list(set([t['itemId'] for t in train_data]))
    user_item_df = pandas.DataFrame(numpy.zeros([len(user_list), len(item_list)]), index=user_list, columns=item_list)

    for i in train_data:
        user_item_df.at[i['userId'], i['itemId']] = i['score']
    print("Time for preparing train data: {}".format(time.time() - startT))
    return user_item_df


def getTestData():
    print("-------- Prepare test data --------")
    startT = time.time()
    test_data = open('test.dat', encoding='utf-8', mode='r')
    test_data = [l.split(' ') for l in test_data.readlines()]
    test_data = pandas.DataFrame(test_data, columns=["userId", "itemId"])
    print("Time for preparing test data: {}".format(time.time() - startT))
    return test_data
