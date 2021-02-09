import argparse
import math
import os
import re
import sys
import time
from datetime import datetime

import nltk
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from DataLoader import *
from utils import sample_sequence

device = torch.device('cuda')


def clean_str(strings):
    new_strings = []
    for string in strings:
        string = re.sub(r' +', ' ', string)
        if len(string.split(' ')) < 6 and len(new_strings) > 0:
            string = new_strings[-1]
        new_strings.append(string)
    return new_strings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt2', type=str)
    parser.add_argument('--do_train', default=False, action="store_true", help="Train the model")
    parser.add_argument('--do_test', default=False, action="store_true",
                        help="Compute bleu-1/2/3 and save the decoded results")
    parser.add_argument('--do_verify', default=False, action="store_true",
                        help="Compute adversarial accuracy score")
    parser.add_argument('--epoch', default=10, type=int,
                        help="Number of epochs to train the model")
    parser.add_argument('--batch_size', default=6, type=int,
                        help="Total examples per table to use")
    parser.add_argument('--window_size', default=15, type=int,
                        help="Size of window for shuffling when creating a batch")
    parser.add_argument('--random_sampling', default=10, type=int,
                        help="Percentage of total data to be used as input")
    parser.add_argument('--learning_rate', default=2e-6, type=float)
    parser.add_argument('--every', default=50, type=int, help="Evaluate after how many steps")
    parser.add_argument('--start_epoch', default=0, type=int, help="Resume from epoch")
    parser.add_argument('--load_from', default='', type=str, help="Load model from this path")
    parser.add_argument('--id', default='models', type=str, help="ID of the experiment")
    parser.add_argument('--max_len', default=800, type=int, help="Max length of the table description")
    parser.add_argument('--decode_first_K', type=int, default=10000, help="For debugging purpose")
    args = parser.parse_args()

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    if args.model == 'gpt2-medium':
        args.batch_size = 2

    print(args)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    tokenizer.add_tokens(['[ENT]', '[SEP]'])

    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    model = nn.DataParallel(model)
    model.to(args.device)

    if not os.path.exists(args.id):
        os.mkdir(args.id)

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    if args.do_train:
        recording_time = datetime.now().strftime('%m_%d_%H_%M')
        tb_writer = SummaryWriter(log_dir='tensorboard/GPT_new_C2F/{}'.format(recording_time))
        dataset = GPTTableCoarseFineDatabase3('data/train_lm_new.json', None, None, tokenizer, args.batch_size,
                                              args.max_len, window_size=args.window_size,
                                              random_sampling=args.random_sampling)
        # if args.stage == 2:
        #     model.load_state_dict(torch.load(args.load_from))

        model.train()
        optimizer = optim.Adam(model.parameters(), args.learning_rate)

        avg_loss = 0
        global_step = 0
        if args.start_epoch != 0:
            checkpoint = torch.load('{}/GPT_new_C2F_ep{}.pt'.format(args.id, args.start_epoch - 1))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for epoch_idx in range(args.start_epoch, args.start_epoch + args.epoch):
            print("start training {}th epoch".format(epoch_idx))
            for idx in tqdm(range(0, dataset.train_len()), total=dataset.train_len()):
                batch = dataset.get_train_data()
                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch
                inputs = torch.cat([caption, trg_inp], 1)

                model.zero_grad()
                optimizer.zero_grad()

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))

                loss = loss * mask.view(-1)
                loss = loss.sum() / mask.sum()

                avg_loss += loss.item()

                loss.backward()
                optimizer.step()
                global_step += 1

                if idx % args.every == 0 and idx > 0:
                    tb_writer.add_scalar("perplexity", math.exp(avg_loss / args.every), global_step)

                    # fake_inputs = inputs
                    # gt_inputs = trg_out.cpu().data.numpy()

                    # samples = sample_sequence(model, 50, fake_inputs, [])
                    # samples = samples[:, caption.shape[1]:]
                    # samples = samples.cpu().data.numpy()
                    #
                    # for s, gt in zip(samples, gt_inputs):
                    #     print("EPOCH {}; FINISHED {}/{}".format(epoch_idx, idx, dataset.train_len()))
                    #     text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    #     text = text[: text.find(tokenizer.eos_token)]
                    #     print("PREDICTION |||||| ", text)
                    #     text = tokenizer.decode(gt, clean_up_tokenization_spaces=True)
                    #     text = text[: text.find(tokenizer.eos_token)]
                    #     print("GROUNDTRUH |||||| ", text)
                    #     break

                    avg_loss = 0

            if args.model == 'gpt2':
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    '{}/GPT_new_C2F_ep{}.pt'.format(args.id, epoch_idx))
            else:
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    '{}/GPT_new_C2F_medium_ep{}.pt'.format(args.id, epoch_idx))

    if args.do_test:
        assert 'stage2' in args.load_from, "The testing can only be done with stage2 model"
        dataset = GPTTableCoarseFineDatabase2(None, None, 'data/test_lm.json', tokenizer, args.batch_size, args.max_len,
                                              args.stage)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        sent_bleus_1 = []
        sent_bleus_2 = []
        sent_bleus_3 = []

        results = {}
        start_time = time.time()
        with torch.no_grad():
            for idx in range(0, min(args.decode_first_K, dataset.test_len())):
                batch = dataset.get_data(idx, 'test')
                references = dataset.get_reference(idx, 'test')
                table_id = dataset.get_table_id(idx, 'test')
                results[table_id] = []

                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch

                fake_inputs = caption

                samples = sample_sequence(model, 50, fake_inputs, [], stop_token=tokenizer.eos_token_id,
                                          top_k=1, trigger=tokenizer.convert_tokens_to_ids('[SEP]'),
                                          supress=[tokenizer.convert_tokens_to_ids('[SEP]'),
                                                   tokenizer.convert_tokens_to_ids('[ENT]')],
                                          repetition=tokenizer.convert_tokens_to_ids('[ENT]'))

                samples = samples[:, caption.shape[1]:]
                samples = samples.cpu().data.numpy()

                intermediate = []
                for s in samples:
                    text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    text = text[text.find('[SEP]') + 6: text.find(tokenizer.eos_token)].strip()
                    # text = text[: text.find(tokenizer.eos_token)]
                    intermediate.append(text)

                results[table_id] = clean_str(intermediate)

                for text in results[table_id]:
                    hypothesis = text.lower().split()
                    sent_bleus_1.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(1, 0, 0)))
                    sent_bleus_2.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.5, 0.5, 0)))
                    sent_bleus_3.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.33, 0.33, 0.33)))

                bleu_1 = format((sum(sent_bleus_1) / len(sent_bleus_1) * 100), '.2f')
                bleu_2 = format((sum(sent_bleus_2) / len(sent_bleus_2) * 100), '.2f')
                bleu_3 = format((sum(sent_bleus_3) / len(sent_bleus_3) * 100), '.2f')

                sys.stdout.write(
                    "finished {}/{}; BLEU score {}/{}/{}; speed={}s/sent \r".format(
                        idx, dataset.test_len(), bleu_1, bleu_2, bleu_3,
                        (time.time() - start_time) / len(sent_bleus_1)))

            print("total corpus BLEU score = {}/{}/{}".format(bleu_1, bleu_2, bleu_3))

        with open('outputs/GPT_{}_C2F_{}.json'.format(args.model, bleu_3), 'w') as f:
            json.dump(results, f, indent=2)

    if args.do_verify:
        assert 'stage2' in args.load_from, "The testing can only be done with stage2 model"
        assert args.stage == 2, "The verification can only be done with stage 2 model"
        dataset = GPTTableCoarseFineDatabase2(None, None, 'data/test_lm_pos_neg.json',
                                              tokenizer, args.batch_size, args.max_len, args.stage)

        model.load_state_dict(torch.load(args.load_from))
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for idx in range(0, dataset.test_len()):
                batch_pos, batch_neg = dataset.get_pair_data(idx, 'test', mask_type='both')

                batch = tuple(Variable(t).to(device) for t in batch_pos)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0][:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
                pos_loss = loss.reshape(logits.shape[0], -1) * mask
                pos_loss_per_instance = pos_loss.sum(1) / mask.sum(1)

                batch = tuple(Variable(t).to(device) for t in batch_neg)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0][:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
                neg_loss = loss.reshape(logits.shape[0], -1) * mask
                neg_loss_per_instance = neg_loss.sum(1) / mask.sum(1)

                comparison = (pos_loss_per_instance < neg_loss_per_instance).float()
                correct += comparison.sum(-1).item()
                total += comparison.shape[0]
                sys.stdout.write('finished {}/{} accuracy {} \r'.format(idx, dataset.test_len(), correct / total))
        print('total accuracy = {}'.format(correct / total))
