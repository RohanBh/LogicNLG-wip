import json
from pathlib import Path
import re
import pprint
import numpy as np

def create_csvs():
    base_dir = Path('data', 'l2t', 'all_csv')
    base_dir.mkdir(parents=True, exist_ok=True)

    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    finished = set()
    for entry in data:
        i = entry['url'].find('all_csv/')
        name = entry['url'][i + 8:]
        if name in finished:
            continue
        with open(base_dir / name, 'w') as fp:
            fp.write('#'.join(entry['table_header']))
            fp.write('\n')
            for row in entry['table_cont']:
                fp.write('#'.join(row))
                fp.write('\n')
        finished.add(name)
    return


def print_all_first_funcs():
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    counter = {}
    for entry in data:
        fn = entry['logic']['func']
        if fn not in counter:
            counter[fn] = 0
        counter[fn] += 1

    total = len(data)
    for k in counter.keys():
        counter[k] = round(100 * counter[k] / total, 2)

    # new_counter = {}
    for k, v in sorted(list(counter.items()), key=lambda x: x[1], reverse=True):
        print(k, ':', v)
        # new_counter[k] = v

    # import pprint
    # pprint.pprint(new_counter)
    return


def print_type_2_func():
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    def recursive_get(logic_json):
        all_funcs = {logic_json['func']}
        for arg in logic_json['args']:
            if type(arg) is dict:
                all_funcs.update(recursive_get(arg))
            else:
                continue
        return all_funcs

    type2set = {}
    for entry in data:
        action = entry['action']
        if action not in type2set:
            type2set[action] = set()
        type2set[action].update(recursive_get(entry['logic']))

    for action in type2set:
        type2set[action] -= set(f for f in type2set[action] if 'filter' in f or 'hop' in f or 'and' in f)
    pprint.pprint(type2set)
    return


def check_ordinal_property():
    """This function checks if there are some ordinal types which don't have a 1st, 2nd
    type of number in their sentence"""
    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    count = 0
    pat = r'\d(th|nd|rd)'
    fn_sents = []
    for entry in data:
        if entry['action'] != 'ordinal':
            continue
        # replace tokens
        sent = entry['sent']
        sent = sent.replace('first', '1st').replace('second', '2nd').replace('third', '3rd').replace(
            'fourth', '4th').replace('fifth', '5th').replace('sixth', '6th').replace('seventh', '7th').replace(
            'eighth', '8th').replace('ninth', '9th')
        if len(re.findall(pat, sent)) == 0:
            count += 1
            fn_sents.append(sent)
    print("Total ordinals with no matches", count, 'out of', len(data))
    print(np.random.choice(fn_sents, 10))
    return


if __name__ == '__main__':
    # print_all_first_funcs()
    print_type_2_func()
    # check_ordinal_property()
