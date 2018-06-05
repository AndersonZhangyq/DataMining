import json
from sklearn import metrics


train_tokens_file = 'train_tokens.json'
train_topics_file = 'train_topics.json'
test_tokens_file = 'test_tokens.json'
output_file = '10152510271.json'


k_docid = 'docid'
k_topic = 'topic'
k_cluster = 'cluster'


def execute_cluster(tokens_list):
    """
    输入一组文档列表，返回每条文档对应的聚类编号
    :param tokens_list: list，每个元素为dict，dict中键值含义如下：
        'docid': int，文档标识符
        'tokenids': list，每个元素为int，单词标识符
    :return: list，每个元素为dict，dict中键值含义如下：
        'docid': int，文档标识符
        'cluster': int，标识该文档的聚类编号
    """

    print(type(tokens_list), type(tokens_list[0]), list(tokens_list[0].items()))    # 仅用于验证数据格式
    #### 修改此处 ####
    import numpy as np
    from copy import deepcopy
    clusters_list = deepcopy(tokens_list)
    # for d in clusters_list:
    #     d['cluster'] = np.random.randint(0, 65)
    from Cluster.TextCluster import TextCluster
    model = TextCluster(clusters_list)
    clusters_list = model.useTFIDF()
    print(type(clusters_list[0]['cluster']))
    print(len(tokens_list), len(clusters_list))

    return clusters_list


""" 以下内容修改无效 """


def calculate_nmi(topics_list, clusters_list):
    id2topic = dict([(d[k_docid], d[k_topic]) for d in topics_list])
    id2cluster = dict([(d[k_docid], d[k_cluster]) for d in clusters_list])
    common_idset = set(id2topic.keys()).intersection(id2cluster.keys())
    if not len(common_idset) == len(topics_list) == len(clusters_list):
        print(len(common_idset), len(topics_list), len(clusters_list))
        print('length inconsistent, result invalid')
        return 0
    else:
        topic_cluster = [(id2topic[docid], id2cluster[docid]) for docid in common_idset]
        y_topic, y_cluster = list(zip(*topic_cluster))
        nmi = metrics.normalized_mutual_info_score(y_topic, y_cluster)
        print('nmi:{}'.format(round(nmi, 4)))
        return nmi


def dump_array(file, array):
    lines = [json.dumps(item, sort_keys=True) + '\n' for item in array]
    with open(file, 'w') as fp:
        fp.writelines(lines)


def load_array(file):
    with open(file, 'r') as fp:
        array = [json.loads(line.strip()) for line in fp.readlines()]
    return array


def clean_clusters_list(clusters_list):
    return [dict([(k, int(d[k])) for k in [k_docid, k_cluster]]) for d in clusters_list]


def evaluate_train_result():
    train_tokens_list = load_array(train_tokens_file)
    train_clusters_list = execute_cluster(train_tokens_list)
    train_topics_list = load_array(train_topics_file)
    calculate_nmi(train_topics_list, train_clusters_list)


def generate_test_result():
    test_tokens_list = load_array(test_tokens_file)
    test_clusters_list = execute_cluster(test_tokens_list)
    test_clusters_list = clean_clusters_list(test_clusters_list)
    dump_array(output_file, test_clusters_list)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    if args.train:
        evaluate_train_result()
    elif args.test:
        generate_test_result()
