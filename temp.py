import json
import time


def count_ent(template):
    return sum(1 for x in template.split(' ') if x == '[ENT]')


def comp_data_len():
    train_name = 'data/train_lm_preprocessed.json'
    start_time = time.time()
    print("Beginning loading data...")
    with open(train_name, 'r') as f:
        train_data = json.load(f)
    print("Done!")
    print(f"Time: {time.time() - start_time}s")

    all_templates = [entry[3] for entry in train_data]
    ent_counts = [count_ent(t) for t in all_templates]

    data_len = sum(2 ** ec for ec in ent_counts)
    print(f"Data length\nOLD: {len(train_data)}, NOW: {data_len}")
    return


if __name__ == '__main__':
    comp_data_len()
    # with open('data/train_lm.json', 'r') as f:
    #     train_data = json.load(f)
    #     print(f"Total tables: {len(list(train_data.keys()))}")
