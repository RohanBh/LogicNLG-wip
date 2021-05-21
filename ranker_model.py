import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup, \
    RobertaForSequenceClassification, RobertaConfig

from program_lstm import NEW_TOKENS


def create_ranker_train_data():
    def get_hash(table_name, sent):
        formatted_sent = re.sub(r"[^\w\s]", '', sent)
        formatted_sent = re.sub(r"\s+", '-', formatted_sent)
        if len(formatted_sent) > 100:
            formatted_sent = formatted_sent[-100:]
        formatted_sent = table_name + '-' + formatted_sent
        return formatted_sent

    def get_prog_set(prog_list):
        # ret_val = prog[prog.rfind('=') + 1:-5]
        prog_list = [prog[:prog.rfind('=')] for prog in prog_list]
        return set(prog_list)

    with open('data/programs_filtered.json') as fp:
        pos_data = json.load(fp)

    with open('data/all_programs.json') as fp:
        all_data = json.load(fp)
    all_data = [x for x in all_data if x is not None]
    all_data = [x for x in all_data if len(x[-1]) > 0]

    pos_data_dict = {}
    for entry in tqdm(pos_data):
        table_name, sent = entry[:2]
        entry_hash = get_hash(table_name, sent)
        pos_data_dict[entry_hash] = entry

    out_data = []
    for entry in tqdm(all_data):
        table_name, sent = entry[:2]
        entry_hash = get_hash(table_name, sent)
        if entry_hash not in pos_data_dict:
            continue
        masked_linked_sent = pos_data_dict[entry_hash][3]
        pos_progs = get_prog_set(pos_data_dict[entry_hash][-1])
        for prog in pos_progs:
            out_data.append((
                table_name, sent, masked_linked_sent, prog, 1
            ))
        all_progs = get_prog_set(entry[-1])
        for prog in all_progs - pos_progs:
            out_data.append((
                table_name, sent, masked_linked_sent, prog, 0
            ))

    with open('data/train_ranker.json', 'w') as fp:
        json.dump(out_data, fp, indent=2)
    return


def get_stats():
    with open('data/train_ranker.json') as fp:
        data = json.load(fp)

    num_pos = len([1 for x in data if x[-1] == 1])
    num_neg = len([1 for x in data if x[-1] == 0])
    assert num_neg + num_pos == len(data)
    print(f"Pos: {num_pos}, Neg: {num_neg}, Total: {len(data)}")
    return


def downsample_neg():
    with open('data/train_ranker.json') as fp:
        data = json.load(fp)
    pos_data = [x for x in data if x[-1] == 1]
    neg_data = [x for x in data if x[-1] == 0]
    ids = np.random.choice(range(len(neg_data)), 100000)
    neg_data = [neg_data[i] for i in ids]
    data = pos_data + neg_data
    random.shuffle(data)
    with open('data/train_ranker.json', 'w') as fp:
        json.dump(data, fp, indent=2)
    return


class RobertaRanker(nn.Module):
    def __init__(self, model_name, device_str='cpu'):
        super(RobertaRanker, self).__init__()
        self.model_name = model_name
        self.device = torch.device(device_str)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        with open('data/logic_form_vocab.json') as fp:
            vocab = json.load(fp)
        new_tokens = sorted(list(vocab['actions'].keys()))
        self.tokenizer.add_tokens(NEW_TOKENS + new_tokens)
        model_config = RobertaConfig.from_pretrained(self.model_name, num_labels=2, return_dict=True)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, config=model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        if device_str != 'cpu':
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor
        return

    def forward(self, padded_sequences, labels):
        if self.device != torch.device('cpu'):
            # (batch_size, seq_len)
            input_ids = padded_sequences['input_ids'].cuda()
            mask = padded_sequences['attention_mask'].cuda()
        else:
            input_ids = padded_sequences['input_ids']
            mask = padded_sequences['attention_mask']
        return self.model(input_ids=input_ids, attention_mask=mask, labels=labels)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            'args': self.model_name,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
        return

    @classmethod
    def load(cls, model_path, cuda=False):
        device_str = 'cuda' if cuda else 'cpu'
        params = torch.load(model_path, map_location=torch.device(device_str))
        model_name = params['args']

        roberta_ranker = cls(model_name, device_str)
        saved_state = params['state_dict']
        roberta_ranker.load_state_dict(saved_state)
        if cuda:
            roberta_ranker = roberta_ranker.cuda()
        roberta_ranker.eval()
        return roberta_ranker

    @staticmethod
    def _get_input_sent(ml_sent, prog, model):
        return prog + ' ' + model.tokenizer.sep_token + ' ' + ml_sent

    @staticmethod
    def train_ranker(args):
        print(args)
        save_path = args.save_base_dir / Path('roberta_ranker_models/')
        save_path.mkdir(exist_ok=True, parents=True)
        device_str = 'cuda' if args.cuda else 'cpu'
        device = torch.device(device_str)
        if args.resume_train:
            model = RobertaRanker.load(args.model_path, args.cuda)
        else:
            model = RobertaRanker(args.model, device_str)
        model.to(device)
        model.train()

        with open('data/train_ranker.json') as fp:
            data = json.load(fp)

        train_data = []
        for entry in data:
            ip_sent = RobertaRanker._get_input_sent(entry[2], entry[3], model)
            train_data.append((ip_sent, entry[-1]))

        optimizer = AdamW(model.parameters(), args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps,
            num_training_steps=args.epochs * (len(train_data) // args.batch_size + 1))

        global_step = avg_loss = 0

        if args.resume_train:
            checkpoint = torch.load(args.ckpt_path)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            recording_time = checkpoint['recording_time']
            global_step = checkpoint['total_steps']
            start_epoch = checkpoint['epochs_finished'] + 1
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            start_epoch = 0
            recording_time = datetime.now().strftime('%m_%d_%H_%M')

        tb_writer = SummaryWriter(log_dir=f'tensorboard/roberta-ranker/{recording_time}')
        print("Recording Time:", recording_time)

        for epoch_idx in range(start_epoch, args.epochs):
            print("start training {}th epoch".format(epoch_idx))
            random.shuffle(train_data)
            true_labels, predict_labels = [], []
            for idx in tqdm(range(0, len(train_data), args.batch_size),
                            total=len(train_data) // args.batch_size + 1):
                global_step += 1
                sent_list, labels = zip(*train_data[idx:idx + args.batch_size])
                padded_sequences = model.tokenizer(sent_list, padding=True, truncation=True, return_tensors="pt")
                labels = model.new_long_tensor(labels).unsqueeze(1)
                optimizer.zero_grad()
                output = model(padded_sequences, labels)
                loss = output.loss
                avg_loss += loss.item()
                logits = output.logits

                true_labels += labels.flatten().tolist()
                predict_labels += logits.argmax(axis=-1).flatten().tolist()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                scheduler.step()

                if idx % args.every == 0 and idx > 0:
                    tb_writer.add_scalar("Avg loss", avg_loss / args.every, global_step)
                    avg_loss = 0

            tb_writer.add_scalar('Train Accuracy',
                                 accuracy_score(true_labels, predict_labels),
                                 epoch_idx)
            prec, recall, f1, _ = precision_recall_fscore_support(true_labels, predict_labels, average='binary')
            tb_writer.add_scalar('Train Precision', prec, epoch_idx)
            tb_writer.add_scalar('Train Recall', recall, epoch_idx)
            tb_writer.add_scalar('Train F1', f1, epoch_idx)

            if (epoch_idx + 1) % args.save_every == 0:
                torch.save({
                    'recording_time': recording_time,
                    'total_steps': global_step,
                    'epochs_finished': epoch_idx,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()},
                    save_path / f'ws_ckpt_{epoch_idx:03}.pt')
                model.save(save_path / f'ws_model_{epoch_idx:03}.pt')

        tb_writer.flush()
        tb_writer.close()
        return


def init_ranker_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', default=False, help='Weakly supervised training')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    parser.add_argument('--model', default='roberta-base', type=str,
                        help="The pretrained model to use as a language model")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', default=8, type=int, help="The batch size to use during training")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning Rate of AdamW")
    parser.add_argument('--warmup_steps', default=2000, type=int, help="Number of steps for increasing lr to max")
    parser.add_argument('--every', default=50, type=int, help="Log after every n examples")
    parser.add_argument('--save_every', default=5, type=int, help="Save after every n epochs")
    parser.add_argument('--resume_train', action='store_true', default=False, help="Resume from save model epoch")
    parser.add_argument('--save_base_dir', default='',
                        type=str, help="Save model and ckpt in this directory")
    parser.add_argument('--model_path', default='', type=str, help="Load model from this path")
    parser.add_argument('--ckpt_path', default='', type=str, help="Load checkpoint from this path")
    return parser.parse_args()


def main():
    # create_ranker_train_data()
    # downsample_neg()
    # get_stats()
    args = init_ranker_arg_parser()
    if args.do_train:
        RobertaRanker.train_ranker(args)
    return


if __name__ == '__main__':
    main()
