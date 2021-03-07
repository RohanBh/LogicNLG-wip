import argparse
import math
import os
import sys
import time
from datetime import datetime

import nltk
import numpy as np
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from DataLoader import *
from Model import ActorCritic
from gen_new_data import get_ent_vals, ent_mask
from utils import sample_sequence_2

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
    parser.add_argument('--do_train', default=False, action="store_true", help="Train a prediction model")
    parser.add_argument('--do_rl', default=False, action="store_true", help="Use RL to train a sequence order model")
    parser.add_argument('--n_actions', default=10, type=int, help="Total maximum actions that the agent can consider")
    parser.add_argument('--episodes', default=10000, type=int, help="Total episodes to train the model")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discounting factor")
    parser.add_argument('--max_steps', default=15, type=int, help="Maximum steps allowed")
    parser.add_argument('--save_every', default=500, type=int, help="save every n episodes")
    parser.add_argument('--start_episode', default=0, type=int, help="The episode to resume training")

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

    if args.do_rl:
        args.batch_size = 1

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

    if args.do_rl:
        # tmm = template masking model
        recording_time = datetime.now().strftime('%m_%d_%H_%M')
        tb_writer = SummaryWriter(log_dir='tensorboard/GPT_tmm/{}'.format(recording_time))

        scorer = model
        scorer.load_state_dict(torch.load(args.load_from)['model_state_dict'])
        scorer.eval()

        actor_critic = ActorCritic(n_actions=args.n_actions, device=args.device)
        actor_critic = nn.DataParallel(actor_critic)
        actor_critic.to(args.device)
        actor_critic.train()

        optimizer = optim.Adam(actor_critic.parameters(), args.learning_rate)

        env = GPTSentenceMaskEnv('data/val_lm_improved.json', tokenizer, scorer, device=args.device)

        best_score = 0
        avg_loss_1 = 0
        avg_loss_2 = 0
        score_history = []
        ep_len_history = []

        if args.start_episode != 0:
            checkpoint = torch.load('{}/GPT_RL_episode_{}.pt'.format(args.id, args.start_episode))
            actor_critic.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for i in tqdm(range(args.start_episode, args.episodes + 1), total=args.episodes):
            state = env.reset()
            done = False
            score = 0
            step_idx = 1
            while not done:
                probs, state_value = actor_critic(state)
                action_probs = torch.distributions.categorical.Categorical(probs)
                action = action_probs.sample()

                next_state, reward, done, _ = env.step(action)
                done = done or step_idx >= args.max_steps
                score += reward

                if not done:
                    _, next_state_value = actor_critic(next_state)
                else:
                    next_state_value = 0

                log_prob = action_probs.log_prob(action)
                delta = reward + args.gamma * next_state_value * (1 - int(done)) - state_value
                actor_loss = -log_prob * delta
                critic_loss = delta ** 2
                avg_loss_1 += actor_loss
                avg_loss_2 += critic_loss
                total_loss = actor_loss + critic_loss

                actor_critic.zero_grad()
                optimizer.zero_grad()

                total_loss.backward()

                optimizer.step()

                if not done:
                    state = next_state
                    step_idx += 1

                if done:
                    tb_writer.add_scalar("ep_return", score, i)
                    tb_writer.add_scalar("ep_len", step_idx, i)

            score_history.append(score)
            ep_len_history.append(step_idx)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score

            if i % 30 == 0:
                tb_writer.add_scalar("avg_reward", avg_score, i)
                tb_writer.add_scalar("actor_loss", avg_loss_1 / 30, i)
                tb_writer.add_scalar("critic_loss", avg_loss_2 / 30, i)
                tb_writer.add_scalar("best_reward", best_score, i)
                tb_writer.add_scalar("avg_ep_len", np.mean(ep_len_history[-100:]), i)
                tqdm.write('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

            if i % args.save_every == 0:
                torch.save({
                    'model_state_dict': actor_critic.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    '{}/GPT_RL_episode_{:05}.pt'.format(args.id, i))


    if args.do_test:
        dataset = GPTTableCoarseFineDatabase3(None, None, 'data/test_lm.json', tokenizer,
                                              args.batch_size, args.max_len,
                                              template_json='data/test_lm_template.json')
        model.load_state_dict(torch.load(args.load_from)['model_state_dict'])
        model.eval()

        sent_bleus_1 = []
        sent_bleus_2 = []
        sent_bleus_3 = []

        results = {}
        # temp_res = {}
        start_time = time.time()
        total_its = min(args.decode_first_K, dataset.test_len())
        error_set = set()
        with torch.no_grad():
            for idx in tqdm(range(0, total_its), total=total_its):
                references = dataset.get_reference(idx, 'test')
                table_id = dataset.get_table_id(idx, 'test')
                results[table_id] = []
                override_templates = None
                # ent_filled = {}

                while True:
                    batch = dataset.get_data(idx, 'test', override_templates=override_templates)
                    tmplts = batch[0]
                    batch = tuple(Variable(t).to(device) for t in batch[1:])
                    trg_inp, trg_out, mask, caption = batch
                    fake_inputs = caption

                    samples = sample_sequence_2(model, 70, fake_inputs, [], stop_token=tokenizer.eos_token_id,
                                                top_k=1, supress=[tokenizer.convert_tokens_to_ids('[SEP]'),
                                                                  tokenizer.convert_tokens_to_ids('[ENT]')])

                    samples = samples[:, caption.shape[1]:]
                    samples = samples.cpu().data.numpy()
                    override_templates = []
                    intermediate = []
                    ent_absent_list = []

                    for b_idx, s in enumerate(samples):
                        text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                        text = text[: text.find(tokenizer.eos_token)].strip()
                        text = clean_str([text])[0]

                        intermediate.append(text)
                        try:
                            ent_list = get_ent_vals(tmplts[b_idx], text)
                        except ValueError as e:
                            ent_list = []
                            error_set.add((str(e), tmplts[b_idx], text))
                        if len(ent_list) == 0:
                            override_templates.append(text)
                            ent_absent_list.append(True)
                            continue
                        # if b_idx not in ent_filled:
                        #     ent_filled[b_idx] = [False] * len(ent_list)
                        # unfilled_ents = [e_idx for e_idx, is_filled in enumerate(ent_filled[b_idx]) if not is_filled]
                        ent_to_fill = np.random.choice(len(ent_list))
                        unfilled_ents = [False] * len(ent_list)
                        unfilled_ents[ent_to_fill] = True
                        # unfilled_ents[ent_to_fill] = True
                        # override_templates.append(ent_mask(tmplts[b_idx], text, unfilled_ents[ent_to_fill]))
                        override_templates.append(ent_mask(tmplts[b_idx], text, unfilled_ents))

                    if all(ent_absent_list):
                        break

                results[table_id] = intermediate

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

                tqdm.write(
                    "finished {}/{}; BLEU score {}/{}/{}; speed={}s/sent \r".format(
                        idx, dataset.test_len(), bleu_1, bleu_2, bleu_3,
                        (time.time() - start_time) / len(sent_bleus_1)))

            print("total corpus BLEU score = {}/{}/{}".format(bleu_1, bleu_2, bleu_3))

        with open('outputs/GPT_new_{}_C2F_{}_res.json'.format(args.model, bleu_3), 'w') as f:
            json.dump(results, f, indent=2)

        with open('outputs/GPT_new_{}_C2F_{}_error.json'.format(args.model, bleu_3), 'w') as f:
            json.dump(list(error_set), f, indent=2)

        # with open('outputs/GPT_new_{}_C2F_{}_tmp.json'.format(args.model, bleu_3), 'w') as f:
        #     json.dump(temp_res, f, indent=2)

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
