import argparse
import collections
import itertools
import sys
import time


def find_frequent_1_itemsets():
    global min_sup, all_frequent_sets, frequent_set, data
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            tmp = data[i][0][j]
            if tmp in frequent_set:
                frequent_set[tmp] += 1
            else:
                frequent_set[tmp] = 1
    #  Filter elements
    frequent_set = {frozenset([k]): v for k, v in frequent_set.items() if v >= min_sup}
    frequent_set = dict(collections.OrderedDict(sorted(frequent_set.items(), key=lambda t: t[0])))


def delete_not_found_in_frequent_1_itemsets():
    global data, frequent_set
    pattern = [list(e)[0] for e in frequent_set.keys()]
    for row in data:
        tmp = row[0]
        tmp = [x for x in tmp if x in pattern]
        if len(tmp) == 0:
            data.remove(row)
        else:
            row[0] = tmp


def count_frequent(pattern_lit):
    global data, frequent_set, min_sup
    frequent_set_count = {}
    for row in data:
        for pattern in pattern_lit:
            if pattern.issubset(set(row[0])):
                row[1] = True
                if pattern in frequent_set_count:
                    frequent_set_count[pattern] += 1
                else:
                    frequent_set_count[pattern] = 1
    return {k: v for k, v in frequent_set_count.items() if v >= min_sup}


def remove_unvisited_row():
    global data
    # for row in data:
    #     if not row[1]:
    #         data.remove(row)
    #     else:
    #         row[1] = False
    data = [[row[0], False] for row in data if row[1] == True]


def hash_function(pattern):
    global bucket_size
    x = pattern[0]
    y = pattern[-1]
    return int((int(x) * 10 + int(y)) % bucket_size)


def hash_improve(length):
    global bucket_size, min_sup
    hash_vector = [0] * bucket_size
    list_ret = [True] * bucket_size
    for row in data:
        row_subsets = itertools.combinations(row[0], length)
        for r_s in row_subsets:
            i = hash_function(r_s)
            if not list_ret[i]:
                continue
            hash_vector[i] += 1
            if hash_vector[i] >= min_sup:
                list_ret[i] = False
    return list_ret


def apriori_gen(length):
    # pre:   length >= 2
    # post:  frequent_set = k-frequency-itemsets (k == length)
    global min_sup, frequent_set
    # cut_for_current_length = hash_improve(length)
    tmp_frequent_set_list = []
    pattern_list = list(frequent_set.keys())
    pattern_list_length = len(pattern_list)
    for i in range(pattern_list_length):
        for j in range(i + 1, pattern_list_length):
            #  Build new pattern
            pattern_1 = pattern_list[i]
            pattern_2 = pattern_list[j]
            if list(pattern_1)[:length - 2] == list(pattern_2)[:length - 2]:
                new_pattern = pattern_1 | pattern_2
                #  0)  if not frequent according to hash tree, drop that
                # if cut_for_current_length[hash_function(list(new_pattern))]:
                #     continue
                #  1)  if subset is not contained, drop that
                should_drop = False
                if length > 2:
                    new_pattern_list = list(new_pattern)
                    for k in new_pattern_list:
                        pattern_test = frozenset([e for e in new_pattern_list if e != k])
                        if pattern_test not in frequent_set:
                            should_drop = True
                            break
                if not should_drop:
                    tmp_frequent_set_list.append(new_pattern)
    #  2)  Check whether new pattern's frequency is larger than min_sup
    frequent_set = count_frequent(tmp_frequent_set_list)


def apriori():
    global min_sup, all_frequent_sets, frequent_set
    # Start mine frequent set
    # Initial Round
    find_frequent_1_itemsets()

    if len(frequent_set) == 0:
        print("No frequent set found!")
        exit()

    delete_not_found_in_frequent_1_itemsets()

    all_frequent_sets.append(frequent_set)

    #  Build k frequent set
    for k in range(2, max_length + 1):
        apriori_gen(k)
        if len(frequent_set) == 0:
            break
        remove_unvisited_row()
        k += 1
        all_frequent_sets.append(frequent_set)


# Arguments to parse
arguments = [["support", "minimum support, range from 0 to 1"],
             ["inputFile", "path to input file"],
             ["outputFile", "path to output file, stdout is default. Create file if not found."]]

# Create parser
parser = argparse.ArgumentParser()
parser.add_argument(arguments[0][0], type=float, help=arguments[0][1])
parser.add_argument(arguments[1][0], help=arguments[1][1])
parser.add_argument(arguments[2][0], help=arguments[2][1], nargs='?', default=sys.stdout)

args = parser.parse_args()

# Parse arguments
support = args.support
if support <= 0 or support >= 1:
    print("Argument Error: support should be 0 to 1!")
    parser.print_help()
    exit()
try:
    inputFile = open(args.inputFile, "r")
except:
    print("File not found error: No such file: {input_file} !".format(input_file=args.inputFile))
    parser.print_help()
    exit()

# Set output
if args.outputFile == sys.stdout:
    outputFile = sys.stdout
else:
    outputFile = open(args.outputFile, "w+")

# Read data
data_str = inputFile.read().strip()
data_split = data_str.split("\n")
transaction_size = len(data_split)
min_sup = transaction_size * support
bucket_size = int(min_sup * .8)

time_sum = 0.0
for _ in range(10):
    data = []
    max_length = 0
    for i in range(len(data_split)):
        # Split data
        tmp = [int(t) for t in data_split[i].split(" ")]
        tmp_length = len(tmp)
        if max_length < tmp_length:
            max_length = tmp_length
        data.append([tmp, False])

    all_frequent_sets = []
    frequent_set = {}

    s_time = time.time()

    apriori()

    e_time = time.time()
    print("Time cost: {cost}".format(cost=(e_time - s_time)))
    time_sum += e_time - s_time

print("Time cost avg: {cost}".format(cost=(time_sum / 10)))

# '''
# # Feed output
for i in all_frequent_sets:
    outputFile.write("{")
    firstRound = True
    for k, v in i.items():
        if not firstRound:
            outputFile.write(",")
        firstRound = False
        outputFile.write("\"" + " ".join([str(e) for e in list(k)]) + "\":" + str(v))
    outputFile.write("}\n")

print("Time cost: {cost}".format(cost=(e_time - s_time)))
# '''
