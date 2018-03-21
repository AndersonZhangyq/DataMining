import argparse
import sys
import time


def find_frequent_1_itemsets():
    global min_sup, all_frequent_sets, frequent_set
    for i in range(len(data)):
        for j in range(len(data[i][0])):
            tmp = data[i][0][j]
            if tmp in frequent_set:
                frequent_set[tmp] += 1
            else:
                frequent_set[tmp] = 1

    #  Filter elements
    frequent_set = {k: v for k, v in frequent_set.items() if v >= min_sup}


def count_frequent(pattern):
    global data
    pattern = [str(p) for p in pattern]
    pattern_length = len(pattern)
    count = 0
    for row in data:
        if sum(p in pattern for p in row[0]) == pattern_length:
            row[1] = True
            count += 1
    return count


def remove_unvisited_row():
    global data
    for row in data:
        if not row[1]:
            data.remove(row)
        else:
            row[1] = False


def apriori_gen(length):
    # pre:   length >= 2
    # post:  frequent_set = k-frequency-itemsets (k == length)
    global min_sup, frequent_set
    tmp_frequent_set = {}
    if length > 2:
        pattern_list = []
        for i in list(frequent_set.keys()):
            pattern_list.append([int(v) for v in list(i.split(' '))])
    else:  # length == 2
        pattern_list = [int(v) for v in frequent_set.keys()]
    for i in pattern_list:
        for j in pattern_list:
            if i == j:
                continue
            #  Build new pattern
            if length == 2 and i < j:
                pattern = [i, j]
                pattern_frequency = count_frequent(pattern)
                if pattern_frequency >= min_sup:
                    tmp_frequent_set[' '.join([str(v) for v in pattern])] = pattern_frequency
            elif length > 2:
                if i[length - 2] < j[length - 2] and i[1:length - 2] == j[1:length - 2]:
                    pattern = i + [j[length - 2]]
                    #  1)  if subset is not contained, drop that
                    should_drop = False
                    for k in pattern:
                        string_pattern = ' '.join([str(v) for v in pattern if v != k])
                        if string_pattern not in frequent_set:
                            should_drop = True
                            break
                    if not should_drop:
                        #  2)  Check whether new pattern's frequency is larger than min_sup
                        pattern_frequency = count_frequent(pattern)
                        if pattern_frequency >= min_sup:
                            tmp_frequent_set[' '.join([str(v) for v in pattern])] = pattern_frequency
    frequent_set = tmp_frequent_set


# Arguments to parse
arguments = [["support", "minimum support, range from 0 to 1"],
             ["variable_2", "variable_2 description"],
             ["inputFile", "path to input file"],
             ["outputFile", "path to output file, stdout is default. Create file if not found."]]

# Create parser
parser = argparse.ArgumentParser()
parser.add_argument(arguments[0][0], type=float, help=arguments[0][1])
parser.add_argument(arguments[1][0], type=float, help=arguments[1][1])
parser.add_argument(arguments[2][0], help=arguments[2][1])
parser.add_argument(arguments[3][0], help=arguments[3][1], nargs='?', default=sys.stdout)

args = parser.parse_args()

# Parse arguments
support = args.support
if support <= 0 or support >= 1:
    print("Argument Error: support should be 0 to 1!")
    parser.print_help()
    exit()
variable_2 = args.variable_2
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
data = []
for i in range(len(data_split)):
    # Split data
    data.append([data_split[i].split(" "), False])

max_length = 0
for i in data:
    tmp_length = len(i[0])
    if max_length < tmp_length:
        max_length = tmp_length

all_frequent_sets = []
frequent_set = {}

s_time = time.time()
# Start mine frequent set
# Initial Round
find_frequent_1_itemsets()

if len(frequent_set) == 0:
    print("No frequent set found!")
    exit()

all_frequent_sets.append(frequent_set)

#  Build k frequent set
for k in range(2, max_length + 1):
    apriori_gen(k)
    if len(frequent_set) == 0:
        break
    remove_unvisited_row()
    k += 1
    all_frequent_sets.append(frequent_set)

e_time = time.time()

# Feed output
for i in all_frequent_sets:
    outputFile.write(str(i) + "\n")

print("Time cost: {cost}".format(cost=(e_time - s_time)))
