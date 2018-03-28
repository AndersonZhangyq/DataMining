import argparse
import ast
import sys


def generate_rules():
    global frequentSets, confidence
    # generate support data dictionary: key: frequent set, value: frequency
    support_data = {}
    for e in frequentSets:
        for k, v in e.items():
            support_data[k] = v
    all_rules = []
    for i in range(1, len(frequentSets)):
        for freq_set in frequentSets[i]:
            condition_elements = [frozenset([e]) for e in freq_set]
            if i == 1:
                calculate_confidence(freq_set, condition_elements, support_data, all_rules)
            else:
                # if more than 1 element, check if sub_rule possible
                generate_sub_rules(freq_set, condition_elements, support_data, all_rules)
    return all_rules


def calculate_confidence(freq_set, condition_elements, support_data, all_rules):
    global confidence
    remain_condition = []
    for cond in condition_elements:
        conf = support_data[freq_set] / support_data[freq_set - cond]
        if conf >= confidence:
            outputFile.write(str(freq_set - cond) + '-->' + str(cond) + 'conf:' + str(conf) + "\n")
            all_rules.append((freq_set - cond, cond, conf))
            remain_condition.append(cond)
    return remain_condition


def generate_sub_rules(freq_set, condition_elements, support_data, all_rules):
    m = len(condition_elements[0])
    # make sure at least one element can be the result
    if len(freq_set) > (m + 1):
        condition = merge(condition_elements, m + 1)
        condition = calculate_confidence(freq_set, condition, support_data, all_rules)
        if len(condition) > 1:
            generate_sub_rules(freq_set, condition, support_data, all_rules)


def merge(data, to_length):
    # merge data to length to_length. First to_length - 2 element should be same for merge
    data_length = len(data)
    merged = []
    for i in range(data_length):
        for j in range(i + 1, data_length):
            data_1 = list(data[i])
            data_2 = list(data[j])
            if data_1[:to_length - 2] == data_2[:to_length - 2]:
                merged.append(data[i] | data[j])
    return merged


# Arguments to parse
arguments = [["confidence", "minimum confidence, range from 0 to 1"],
             ["frequentSetFile", "path to frequent set file"],
             ["outputFile", "path to output file, stdout is default. Create file if not found."]]

# Create parser
parser = argparse.ArgumentParser()
parser.add_argument(arguments[0][0], type=float, help=arguments[0][1])
parser.add_argument(arguments[1][0], help=arguments[1][1])
parser.add_argument(arguments[2][0], help=arguments[2][1], nargs='?', default=sys.stdout)

args = parser.parse_args()

# Parse arguments
confidence = args.confidence
if confidence <= 0 or confidence >= 1:
    print("Argument Error: confidence should be 0 to 1!")
    parser.print_help()
    exit()


try:
    frequentSetFile = open(args.frequentSetFile, "r")
except:
    print("File not found error: No such file: {frequentSetFile} !".format(frequentSetFile=args.frequentSetFile))
    parser.print_help()
    exit()

# Set output
if args.outputFile == sys.stdout:
    outputFile = sys.stdout
else:
    outputFile = open(args.outputFile, "w+")

st = frequentSetFile.read().strip()
frequentSets = []
for s in st.split("\n"):
    dict_test = ast.literal_eval(s)
    frequentSets.append({frozenset(k.split(" ")): v for k, v in dict_test.items()})

generate_rules()
