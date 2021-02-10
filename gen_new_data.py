import copy
import json

import numpy as np
from tqdm.auto import tqdm

from utils import powerset


def count_ent_1(template):
    return sum(1 for x in template.split(' ') if x == '[ENT]')


def comp_data_len_1():
    train_name = 'data/train_lm_preprocessed.json'
    with open(train_name, 'r') as f:
        train_data = json.load(f)

    all_templates = [entry[3] for entry in train_data]
    ent_counts = [count_ent_1(t) for t in all_templates]

    data_len = sum(2 ** ec for ec in ent_counts)
    print(f"Data length\nOLD: {len(train_data)}, NOW: {data_len}")
    return


def count_ent_2(template):
    last_ent = False
    total_ents = 0
    for x in template.split(' '):
        if x == '[ENT]':
            if not last_ent:
                total_ents += 1
            last_ent = True
        elif x.isnumeric():
            total_ents += 1
            last_ent = True
        else:
            last_ent = False
    return total_ents


def comp_data_len_2():
    train_name = 'data/train_lm_preprocessed.json'
    with open(train_name, 'r') as f:
        train_data = json.load(f)

    all_templates = [entry[3] for entry in train_data]
    ent_counts = [count_ent_2(t) for t in all_templates]

    data_len = sum(2 ** ec for ec in ent_counts)
    print(f"Data length\nOLD: {len(train_data)}, NOW: {data_len}")
    return


# def _get_ent_vals(yt, y):
#     yts = yt.split(' ')
#     ys = y.split(' ')
#     ent_list = []
#     i2 = 0
#     for i1 in range(len(yts)):
#         if yts[i1] == '[ENT]':
#             ent_list.append([])
#             nw_1 = yts[i1 + 1] if i1 + 1 < len(yts) else '[NO_TOKEN]'
#             nw_2 = ys[i2] if i2 < len(ys) else '[NO_TOKEN]'
#             while nw_1 != nw_2:
#                 ent_list[-1].append(ys[i2])
#                 i2 += 1
#                 nw_2 = ys[i2] if i2 < len(ys) else '[NO_TOKEN]'
#         elif yts[i1] == ys[i2]:
#             if yts[i1].isnumeric():
#                 ent_list.append([yts[i1]])
#             i2 += 1
#             continue
#         else:
#             raise ValueError(f"Wrong input:\nTemplate: {yt},\nTrue: {y}")
#     return ent_list


def get_ent_vals(yt, y):
    yt += '[SPECIAL_EOS]'
    y += '[SPECIAL_EOS]'
    yt = yt.replace(' ,', ',')
    y = y.replace(' ,', ',')

    ent_list = []

    y_start = 0
    all_pos = []
    for sub in yt.split('[ENT]'):
        tmp = y.find(sub, y_start)
        if tmp == -1:
            raise ValueError(f"Wrong input:\nTemplate: {yt},\nTrue: {y}")
        all_pos.append((tmp, tmp + len(sub)))
        y_start += len(sub)

    for i, j in zip(all_pos[:-1], all_pos[1:]):
        ent_list.append(y[i[1]:j[0]])

    return ent_list


def ent_mask(yt, y, mask):
    ent_list = get_ent_vals(yt, y)

    new_y = []
    ent_ix = 0
    for w in yt.split(' '):
        if w == '[ENT]':
            new_y.append(ent_list[ent_ix] if mask[ent_ix] else w)
            ent_ix += 1
        else:
            new_y.append(w)
    return ' '.join(new_y)


# def improve_yt(yt, transform_numeric=False):
#     last_ent = False
#     new_yt_split = []
#     for x in yt.split(' '):
#         if x == '[ENT]':
#             if not last_ent:
#                 new_yt_split.append('[ENT]')
#             last_ent = True
#         elif transform_numeric and x.isnumeric():
#             new_yt_split.append('[ENT]')
#             last_ent = True
#         else:
#             new_yt_split.append(x)
#             last_ent = False
#     return ' '.join(new_yt_split)


def improve_yt(yt):
    last_ent = False
    new_yt_split = []
    for x in yt.split(' '):
        if x == '[ENT]':
            if not last_ent:
                new_yt_split.append('[ENT]')
            last_ent = True
        elif x.isnumeric():
            if not last_ent:
                new_yt_split.append('[ENT]')
            last_ent = True
        else:
            new_yt_split.append(x)
            last_ent = False
    return ' '.join(new_yt_split)


def _duplicate_entry(entry):
    yt = entry[3]
    y = entry[0]

    yt = improve_yt(yt)
    ent_list = get_ent_vals(yt, y)
    total_ents = len(ent_list)

    split_yt = yt.split(' ')
    new_yt_list = []
    for non_masked_ents in powerset(range(total_ents)):
        if len(non_masked_ents) == total_ents:
            continue

        new_split_yt = []
        curr_ent, i = 0, 0

        for w in split_yt:
            if w != '[ENT]':
                new_split_yt.append(w)
                continue
            if curr_ent >= total_ents or i >= len(non_masked_ents) or curr_ent != non_masked_ents[i]:
                new_split_yt.append(w)
            else:
                new_split_yt.append(ent_list[curr_ent])
                i += 1
            curr_ent += 1

        new_yt_list.append(' '.join(new_split_yt))

    entries = []
    for new_yt in new_yt_list:
        new_entry = copy.copy(entry)
        new_entry[3] = new_yt
        entries.append(new_entry)

    return entries


def create_new_json():
    train_name = 'data/train_lm_preprocessed.json'
    with open(train_name, 'r') as f:
        train_data = json.load(f)

    new_train_data = [_duplicate_entry(e) for e in tqdm(train_data)]
    new_train_data = [e for e_l in tqdm(new_train_data) for e in e_l]

    with open('data/train_lm_new.json', 'w') as f:
        json.dump(new_train_data, f, indent=2)

    return


def ent_2_stats():
    train_name = 'data/train_lm_new.json'
    with open(train_name, 'r') as f:
        train_data = json.load(f)

    ent_dist = []
    for entry in tqdm(train_data):
        template = entry[3]
        ent_dist.append(count_ent_1(template))

    tqdm.write(f"#ENT Percentiles: {np.percentile(ent_dist, [0, 25, 50, 75, 80, 95, 100])}")

    ent_len_dist = []
    for entry in tqdm(train_data):
        template = entry[3]
        ent_len_dist.append(len(template.split(' ')))
    tqdm.write(f"Sentence length Percentiles: {np.percentile(ent_len_dist, [0, 25, 50, 75, 80, 95, 100])}")
    tqdm.write(f"mean {np.mean(ent_len_dist)}")
    return


if __name__ == '__main__':
    # comp_data_len_1()
    # comp_data_len_2()
    # create_new_json()

    # with open('data/train_lm.json', 'r') as f:
    #     train_data = json.load(f)
    #     print(f"Total tables: {len(list(train_data.keys()))}")

    # with open('data/train_lm_new.json', 'r') as f:
    #     train_data = json.load(f)
    #     print(f"Total Entries: {len(train_data)}")

    ent_2_stats()
