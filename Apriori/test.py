def delete_not_found_in_frequent_1_itemsets():
    global data, frequent_set
    pattern = [e[0] for e in frequent_set.items()]
    for row in data:
        tmp = row[0]
        tmp = [x for x in tmp if x in pattern]
        if len(tmp) == 0:
            data.remove(row)
        else:
            row[0] = tmp
