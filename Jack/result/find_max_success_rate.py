import re
from os import listdir


def get_max_rate(file_name):
    # get dict
    rate_dict = dict()
    max_rate = 0.0
    with open(file_name, 'r') as handle:
        line_number = 1
        for line in handle:
            # test_data Success Rate = 0.0
            if re.match(r'^test_data Success Rate = ', line):
                rate = re.sub(r'^test_data Success Rate = ', '', line)
                rate_dict[int(line_number)] = float(rate)
                max_rate = max(max_rate, float(rate))
            line_number += 1
    # print(rate_dict)
    # print(max_rate)
    # remove keys
    return_dict = rate_dict.copy()
    for key in rate_dict.keys():
        if rate_dict[key] < max_rate:
            del return_dict[key]
    # print(return_dict)
    return return_dict


all_file = listdir()
for file_name in all_file:
    if re.match(r'^.*.txt', file_name):
        print(file_name)
        print(get_max_rate(file_name))
        print()
