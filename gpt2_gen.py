import argparse
import copy
import json
import random
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

from DataLoader import Dataloader
from l2t_parse_fill import Parser
from program_lstm import _compare


def create_out_sent(masked_ls, mapping, cols, masked_val, recover_dict):
    new_sent = []
    all_types = {'<ENTITY', '<NONENTITY', '<NARG', '<COMPUTE', '<NONLINKED', '<COUNT'}

    for tok in masked_ls.split():
        if tok[0] != '<' or tok[-1] != '>' or not any(t in tok for t in all_types):
            new_sent.append(recover_dict.get(tok, tok))
            continue
        v, (i, j) = mapping[tok]
        v = str(v)
        # -5 nonlink, -4 count, -3 compute, -7 nth, -1 title
        if i in [-5, -4]:
            comp_msk_val_0 = 'msk_nonlink_num' if i == -5 else 'msk_input'
            if masked_val is not None and masked_val[0] == comp_msk_val_0 and _compare(v, masked_val[1]):
                new_sent.append('[MASK]')
            else:
                new_sent.extend(recover_dict.get(_v, _v) for _v in v.split())
        elif i == -3 and j == -3:
            if masked_val is not None and _compare(v, str(masked_val[1])):
                new_sent.append('[MASK]')
            else:
                new_sent.extend(recover_dict.get(_v, _v) for _v in v.split())
        elif i in [-7, -1, 0]:
            new_sent.extend(recover_dict.get(_v, _v) for _v in v.split())
        else:
            # -3, normal
            if masked_val is not None and cols[j] == masked_val[0][4:] and _compare(v, str(masked_val[1])):
                new_sent.append('[MASK]')
            else:
                new_sent.extend(recover_dict.get(_v, _v) for _v in v.split())

    new_sent = ' '.join(new_sent)
    return new_sent


def create_train_data():
    with open('data/programs_filtered.json') as fp:
        data = json.load(fp)

    parser = Parser("data/l2t/all_csv")

    new_data = []
    for entry in tqdm(data):
        table_name, og_sent = entry[:2]
        masked_sent, mapping, rec_dict_sent, all_cols = parser.fake_parse_2(table_name, og_sent)
        table = pd.read_csv(f'data/l2t/all_csv/{entry[0]}', delimiter="#")
        cols = table.columns.tolist()
        sent = create_out_sent(masked_sent, mapping, cols, entry[4], rec_dict_sent)
        all_cols = [idx for idx, col in enumerate(cols) if col in all_cols]
        title = parser.title_mapping[table_name]

        table_summary = ""
        for i in range(len(table)):
            table_summary += 'In row {} , '.format(i + 1)
            for col_idx in all_cols:
                if isinstance(table.iloc[i, col_idx], str):
                    entity = map(lambda x: x.capitalize(), table.iloc[i, col_idx].split(' '))
                    entity = ' '.join(entity)
                else:
                    entity = str(table.iloc[i, col_idx])

                table_summary += 'the {} is {} , '.format(cols[col_idx], entity)
            table_summary = table_summary[:-3] + ' . '

        new_data.append((
            table_name, title, og_sent, all_cols, entry[4], sent, table_summary
        ))

    with open('data/train_gpt2_lm.json', 'w') as fp:
        json.dump(new_data, fp, indent=2)
    return


class GPT2MaskSentDataset(Dataloader):
    def __init__(self, train_name, tokenizer, batch_size=5, max_len=800, window_size=15):
        super(GPT2MaskSentDataset, self).__init__(None, None, None)
        if train_name:
            with open(train_name, 'r') as f:
                self.train = json.load(f)

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.window_size = window_size

    @property
    def train_len(self):
        return len(self.train) // self.batch_size

    def get_train_data(self):
        idx = random.choice(range(0, len(self.train)))

        window = self.window_size  # int(self.batch_size / 2)
        start_idx = max(0, idx - window)
        end_idx = min(idx + window, len(self.train))

        entries = copy.copy(self.train[start_idx: end_idx])
        random.shuffle(entries)
        entries = entries[:self.batch_size]

        seqs = []
        descs = []
        seq_masks = []
        for e in entries:
            seqs.append(self.tokenizer.encode(e[4], add_special_tokens=False))
            seq_masks.append([1] * len(seqs[-1]))
            table_summary = e[-1]

            tsum_ids = self.tokenizer.tokenize(table_summary)
            if len(tsum_ids) > self.max_len:
                tsum_ids = tsum_ids[:self.max_len]

            tsum_prefix = self.tokenizer.tokenize('Given the table title of "{}" . '.format(e[1]))

            descs.append(self.tokenizer.convert_tokens_to_ids(tsum_prefix + tsum_ids))

        length = max(len(_) for _ in seqs) + 1

        for i in range(len(seqs)):
            seqs[i] += (length - len(seqs[i])) * [self.tokenizer.pad_token_id]
            seq_masks[i] = seq_masks[i] + [1] + (length - len(seq_masks[i]) - 1) * [0]
        seqs = torch.LongTensor(seqs)
        seq_masks = torch.FloatTensor(seq_masks)

        length = max([len(_) for _ in descs]) + 1
        for i in range(len(descs)):
            descs[i] = (length - len(descs[i])) * [self.tokenizer.pad_token_id] + descs[i]
        descs = torch.LongTensor(descs)

        inputs = seqs[:, :-1]
        outputs = seqs

        return inputs, outputs, seq_masks, descs

    def get_test_data(self, idx):
        table_id, entry = self.obtain_idx(idx, 'test')
        table = pd.read_csv('data/l2t/all_csv/' + table_id, '#')
        cols = table.columns

        seqs = []
        descs = []
        seq_masks = []

        for e in entry:
            seqs.append(self.tokenizer.encode(e[3], add_special_tokens=False))
            seq_masks.append([1] * len(seqs[-1]))

            table_summary = ""
            for i in range(len(table)):
                table_summary += 'In row {} , '.format(i + 1)
                for col_idx in e[1]:
                    if isinstance(table.iloc[i][cols[col_idx]], str):
                        entity = map(lambda x: x.capitalize(), table.iloc[i, cols[col_idx]].split(' '))
                        entity = ' '.join(entity)
                    else:
                        entity = str(table.iloc[i, cols[col_idx]])

                    table_summary += 'the {} is {} , '.format(cols[col_idx], entity)
                table_summary = table_summary[:-3] + ' . '

            t_sum_toks = self.tokenizer.tokenize(table_summary)
            if len(t_sum_toks) > self.max_len:
                t_sum_toks = t_sum_toks[:self.max_len]

            t_sum_pref_toks = self.tokenizer.tokenize('Given the table title of "{}" . '.format(e[2]))

            descs.append(self.tokenizer.convert_tokens_to_ids(t_sum_pref_toks + t_sum_toks))

        length = max([len(_) for _ in seqs]) + 1

        for i in range(len(seqs)):
            seqs[i] += (length - len(seqs[i])) * [self.tokenizer.pad_token_id]
            seq_masks[i] = seq_masks[i] + [1] + (length - len(seq_masks[i]) - 1) * [0]
        seqs = torch.LongTensor(seqs)
        seq_masks = torch.FloatTensor(seq_masks)

        length = max([len(_) for _ in descs]) + 1
        for i in range(len(descs)):
            descs[i] = (length - len(descs[i])) * [self.tokenizer.pad_token_id] + descs[i]
        descs = torch.LongTensor(descs)

        inputs = seqs[:, :-1]
        outputs = seqs

        return inputs, outputs, seq_masks, descs


class GPT2LM(nn.Module):
    def __init__(self, model_name, device_str='cpu'):
        super(GPT2LM, self).__init__()
        self.model_name = model_name
        self.device = torch.device(device_str)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_tokens(['[MASK]'])
        if 'gpt2' in self.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def forward(self, inputs):
        return self.model(inputs)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            'args': (self.model_name),
            'state_dict': self.state_dict()
        }
        torch.save(params, path)
        return

    @classmethod
    def load(cls, model_path, cuda=False):
        device_str = 'cuda' if cuda else 'cpu'
        params = torch.load(model_path, map_location=torch.device(device_str))
        model_name = params['args']

        gpt2_lm = cls(model_name, device_str)
        saved_state = params['state_dict']
        gpt2_lm.load_state_dict(saved_state)
        if cuda:
            gpt2_lm = gpt2_lm.cuda()
        gpt2_lm.eval()
        return gpt2_lm

    @staticmethod
    def train_lm(args):
        print(args)
        save_path = args.save_base_dir / Path('gpt2_lm_save/')
        save_path.mkdir(exist_ok=True, parents=True)
        device_str = 'cuda' if args.cuda else 'cpu'
        device = torch.device(device_str)
        if args.resume_train:
            model = GPT2LM.load(args.model_path, args.cuda)
        else:
            model = GPT2LM(args.model, device_str)
        model.to(device)
        model.train()

        dataset = GPT2MaskSentDataset('data/train_gpt2_lm.json', model.tokenizer,
                                      args.batch_size, args.max_len, window_size=50)

        optimizer = AdamW(model.parameters(), args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps,
            num_training_steps=dataset.train_len * args.epochs)

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

        tb_writer = SummaryWriter(log_dir=f'tensorboard/gpt2-lm/{recording_time}')
        print("Recording Time:", recording_time)

        for epoch_idx in range(start_epoch, args.epochs):
            print("start training {}th epoch".format(epoch_idx))
            for idx in tqdm(range(dataset.train_len), total=dataset.train_len):
                batch = dataset.get_train_data()
                batch = tuple(t.to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch
                inputs = torch.cat([caption, trg_inp], 1)

                optimizer.zero_grad()

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = model.criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
                loss = loss * mask.view(-1)
                loss = loss.sum() / mask.sum()
                avg_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                scheduler.step()
                global_step += 1

                if idx % args.every == 0 and idx > 0:
                    tb_writer.add_scalar("Avg loss", avg_loss / args.every, global_step)
                    avg_loss = 0

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

    @staticmethod
    def sample_sequences(model, max_len, captions, stop_token, mask_token, device_str='cpu'):
        if isinstance(captions, list):
            captions = torch.tensor(captions, dtype=torch.long, device=device_str)
            # captions = captions.unsqueeze(0).repeat(num_samples, 1)
            captions = captions.unsqueeze(0)

        generated = captions
        batch_size = generated.shape[0]

        is_mask_gen = [False for _ in range(batch_size)]
        finished_sentence = [False for _ in range(batch_size)]

        with torch.no_grad():
            for _ in range(max_len):
                outputs = model(generated)
                next_token_logits = outputs[:, -1, :]

                # Once a [MASK] is generated, don't generate it again
                for b in range(batch_size):
                    if is_mask_gen[b]:
                        next_token_logits[b, mask_token] = -float('Inf')

                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                for b in range(batch_size):
                    if next_token[b].item() == mask_token:
                        is_mask_gen[b] = True
                    if next_token[b].item() == stop_token:
                        finished_sentence[b] = True

                generated = torch.cat((generated, next_token), dim=1)

                if all(finished_sentence):
                    break

        return generated

    @staticmethod
    def _clean_str(strs):
        new_strings = []
        for string in strs:
            string = re.sub(r' +', ' ', string)
            if len(string.split(' ')) < 6 and len(new_strings) > 0:
                string = new_strings[-1]
            new_strings.append(string)
        return new_strings

    @staticmethod
    def test_program_lstm(args):
        print(args)
        save_path = Path('gpt2_lm_outputs/')
        save_path.mkdir(exist_ok=True)
        device_str = 'cuda' if args.cuda else 'cpu'
        device = torch.device(device_str)

        model = GPT2LM.load(args.model_path, args.cuda)
        model.to(device)
        model.eval()

        dataset = GPT2MaskSentDataset('data/test_gpt2_lm.json', model.tokenizer,
                                      args.batch_size, args.max_len, window_size=50)

        with torch.no_grad():
            for idx in tqdm(range(dataset.test_len()), total=dataset.test_len()):
                batch = dataset.get_test_data(idx)
                batch = tuple(t.to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch
                samples = GPT2LM.sample_sequences(model, 50, caption, model.tokenizer.eos_token_id,
                                                  model.tokenizer.convert_tokens_to_ids('[MASK]'))
                samples = samples[:, caption.shape[1]:]
                samples = samples.cpu().data.numpy()

                intermediate = []
                for s in samples:
                    text = model.tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    text = text[:text.find(model.tokenizer.eos_token)].strip()
                    intermediate.append(text)

                result = GPT2LM._clean_str(intermediate)
                # TODO: You have a list of shape (batch,) of masked statements. Now, call the p-lstm to generate program
                #  and store its output with the masked sentence in a json file.

        return


def init_lm_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    parser.add_argument('--model', default='gpt2', type=str,
                        help="The pretrained model to use as a language model")
    parser.add_argument('--epochs', default=10, type=int, help="Number of epochs to train the model")
    parser.add_argument('--batch_size', default=8, type=int, help="The batch size to use during training")
    parser.add_argument('--max_len', default=800, type=int, help="Maximum tokens to use for table summary in the input")
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


if __name__ == '__main__':
    # create_train_data()
    args = init_lm_arg_parser()
    GPT2LM.train_lm(args)
