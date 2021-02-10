import argparse
import time

from torch import nn
from torch.autograd import Variable
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from DataLoader import *
from gen_new_data import improve_yt
from utils import sample_sequence

device = torch.device('cuda')

# def clean_str(strings):
#     new_strings = []
#     for string in strings:
#         string = re.sub(r' +', ' ', string)
#         if len(string.split(' ')) < 6 and len(new_strings) > 0:
#             string = new_strings[-1]
#         new_strings.append(string)
#     return new_strings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt2', type=str)
    parser.add_argument('--do_test', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--epoch', default=10, type=int,
                        help="whether to train or test the model")
    parser.add_argument('--batch_size', default=6, type=int,
                        help="whether to train or test the model")
    parser.add_argument('--learning_rate', default=2e-6, type=float,
                        help="whether to train or test the model")
    parser.add_argument('--every', default=50, type=int, help="evaluate how many steps")
    parser.add_argument('--load_from', default='', type=str, help="load model path")
    parser.add_argument('--max_len', default=800, type=int, help="whether to train or test the model")
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

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    if args.do_test:
        dataset = GPTTableCoarseFineDatabase2(
            None, None, 'data/test_lm.json', tokenizer, args.batch_size, args.max_len, 2)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        results = {}
        start_time = time.time()
        with torch.no_grad():
            total_it = min(args.decode_first_K, dataset.test_len())
            for idx in tqdm(range(0, total_it), total=total_it):
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
                    text = text[:text.find('[SEP]')].strip()
                    # text = text[: text.find(tokenizer.eos_token)]
                    intermediate.append(improve_yt(text))

                # TODO: try what happens with clean_str
                results[table_id] = intermediate
                # results[table_id] = clean_str(results[table_id])

        with open('data/test_lm_template.json'.format(args.model), 'w') as f:
            json.dump(results, f, indent=2)
