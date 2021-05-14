import json
import pprint
from collections import Counter

import matplotlib.pyplot as plt

from utils import plotCDF


def print_succ_masks(data):
    tot_msks = len([x for x in data if x is not None])
    print(f"Total statements: {len(data)}, masked: {tot_msks}, unmasked: {len(data) - tot_msks}")
    return


def print_coverage(data):
    tot_stmts = len([x for x in data if x is not None])
    total_progs = 0
    for entry in data:
        if entry is None:
            continue
        progs = entry[-1]
        if len(progs) > 0:
            total_progs += 1
    print(f'Masked stmts: {tot_stmts}, Progs found: {total_progs}, Coverage: {total_progs * 100 / tot_stmts}')
    return


def show_dist(data):
    counter = {}
    for entry in data:
        if entry is None:
            continue
        progs = entry[-1]
        for prog in progs:
            prog_type = prog.split('{')[0].strip()
            if prog_type == 'hop':
                prog_type = prog.split('{')[1].strip()
            if prog_type not in counter:
                counter[prog_type] = 0
            counter[prog_type] += 1
    print("Program types:")
    new_counter = dict(sorted(counter.items(), key=lambda x: (-x[-1])))
    pprint.pprint(new_counter, indent=2, sort_dicts=False)
    return


def num_progs_cdf(data, is_log=False):
    tot_progs_arr = []
    for entry in data:
        if entry is None:
            continue
        progs = entry[-1]
        if len(progs) > 200:
            print(entry[:2])
        tot_progs_arr.append(len(progs))
    ax = plt.gca()
    plotCDF(ax, {0: tot_progs_arr}, {0: ''}, 'Total Programs', '# of instances', isLog=is_log)
    locs = plt.yticks()[0]
    plt.yticks(locs, [f'{float(f) * len(tot_progs_arr):.0f}' for f in locs])
    print("Max programs:", sorted(tot_progs_arr)[-10:])
    print_dict = dict(sorted(Counter(tot_progs_arr).items(), key=lambda x: (x[-1], x[0]))[-10:])
    print(print_dict)
    plt.show()
    return


def main():
    with open("data/programs.json", 'r') as f:
        data = json.load(f)
    print_succ_masks(data)
    print_coverage(data)
    show_dist(data)
    num_progs_cdf(data, is_log=True)
    print("#" * 100)
    with open("data/programs_filtered.json", 'r') as f:
        data = json.load(f)
    print_succ_masks(data)
    print_coverage(data)
    show_dist(data)
    num_progs_cdf(data)
    return


# Amalyze model outputs.
def print_check_proportion(fname):
    with open(f"plstm_outputs/{fname}", 'r') as f:
        data = json.load(f)
    data = [(x[-1], x[-2] != None) for x in data]
    ctr = Counter(data)
    ctr2 = {
        'FN': ctr[(False, False)],
        'FP': ctr[(False, True)],
        'TP': ctr[(True, True)],
    }
    print(ctr2)
    assert len(data) == sum(ctr2.values())
    print("Coverage (TP / (TP + FP + FN)):", ctr2['TP'] / len(data))
    print("Recall (TP / (TP + FN)):", ctr2['TP'] / (ctr2['TP'] + ctr2['FN']))
    print("Precision (TP / (TP + FP)):", ctr2['TP'] / (ctr2['TP'] + ctr2['FP']))
    return


def show_dist_2(fname):
    with open(f"plstm_outputs/{fname}", 'r') as f:
        data = json.load(f)
    counter = {'none': 0}
    tot_prog_dict = {'none': 0}
    for entry in data:
        prog = entry[-2]
        is_accept = entry[-1]
        if prog is None:
            counter['none'] += 1
            tot_prog_dict['none'] += 1
            continue
        prog_type = prog.split('{')[0].strip()
        if prog_type == 'hop':
            prog_type = prog.split('{')[1].strip()
            if '=' in prog_type and '/False' in prog_type:
                prog_type = 'hop'
        if prog_type not in counter:
            counter[prog_type] = 0
        if prog_type not in tot_prog_dict:
            tot_prog_dict[prog_type] = 0
        if is_accept:
            counter[prog_type] += 1
        tot_prog_dict[prog_type] += 1

    counter = {k: (v, tot_prog_dict[k]) for k, v in counter.items()}
    new_counter = dict(sorted(counter.items(), key=lambda x: x[-1][1], reverse=True))
    pprint.pprint(new_counter, indent=2, sort_dicts=False)
    return


def main2():
    fname = 'out_rb_079.json'
    print_check_proportion(fname)
    show_dist_2(fname)
    return


if __name__ == '__main__':
    # main()
    main2()
