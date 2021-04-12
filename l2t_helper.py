import json
from pathlib import Path


def create_csvs():
    base_dir = Path('data', 'l2t', 'all_csv')
    base_dir.mkdir(parents=True, exist_ok=True)

    with open('data/l2t/all_data.json', 'r') as f:
        data = json.load(f)

    finished = set()
    for entry in data:
        i = entry['url'].find('all_csv/')
        name = entry['url'][i+8:]
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


if __name__ == '__main__':
    print_all_first_funcs()
