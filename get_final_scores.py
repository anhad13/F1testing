import data_ptb as data
import nltk
import glob
from src.utils import config
from src.models import build_model
from src.utils.utils import assert_for_log, maybe_make_dir, load_model_state
from src.preprocess import build_tasks
import os
from allennlp.data.token_indexers import \
    SingleIdTokenIndexer, ELMoTokenCharactersIndexer, \
    TokenCharactersIndexer
from allennlp.data import Vocabulary
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy
import argparse
import re
import sys
sys.path.append("../rjiant/jiant/")

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../../"))
word_tags = [
    'CC',
    'CD',
    'DT',
    'EX',
    'FW',
    'IN',
    'JJ',
    'JJR',
    'JJS',
    'LS',
    'MD',
    'NN',
    'NNS',
    'NNP',
    'NNPS',
    'PDT',
    'POS',
    'PRP',
    'PRP$',
    'RB',
    'RBR',
    'RBS',
    'RP',
    'SYM',
    'TO',
    'UH',
    'VB',
    'VBD',
    'VBG',
    'VBN',
    'VBP',
    'VBZ',
    'WDT',
    'WP',
    'WP$',
    'WRB']
criterion = nn.CrossEntropyLoss()


def evaluate(data_source, batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output = model.decoder(output)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def corpus2idx(sentence):
    arr = np.array([data.dictionary.word2idx[c] for c in sentence.split()], dtype=np.int32)
    return torch.from_numpy(arr[:, None]).long()


# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def MRG(tr):
    if isinstance(tr, str):
        # return '(' + tr + ')'
        return tr + ' '
    else:
        s = '( '
        for subtr in tr:
            s += MRG(subtr)
        s += ') '
        return s


def MRG_labeled(tr):
    if isinstance(tr, nltk.Tree):
        if tr.label() in word_tags:
            return tr.leaves()[0] + ' '
        else:
            s = '(%s ' % (re.split(r'[-=]', tr.label())[0])
            for subtr in tr:
                s += MRG_labeled(subtr)
            s += ') '
            return s
    else:
        return ''


def mean(x):
    return sum(x) / len(x)


if __name__ == '__main__':
    marks = [' ', '-', '=']

    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)
    parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    parser.add_argument('--data', type=str, default='data/penn',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='PTB.pt',
                        help='onlstmconfig/test1_15_75/model_state_main_epoch_269.best_macro.th')
    parser.add_argument('--exp_dir', type=str, default='PTB.pt',
                        help='model checkpoint to use')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--wsj10', action='store_true',
                        help='use WSJ10')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='epochs div to pick checkpoints from.')
    parser.add_argument('--data_dir', type=str, default="data/",
                        help='reload tasks')

    parser.add_argument('--max_targ_word_v_size', type=int, default=20000,
                        help='maxt')

    parser.add_argument('--config_file', type=str, default='',
                        help='maxt')
    parser.add_argument(
        '--ptb_path',
        type=str,
        default='/Users/anhadmohananey/Downloads/ptb_sec23.jsonl')
    parser.add_argument('--use_PP', type=bool, default=False)
    args = parser.parse_args()
    clargs = config.params_from_file(args.config_file, None)
    torch.manual_seed(args.seed)
    pretrain_tasks, target_tasks, vocab, word_embs = build_tasks(clargs)
    tasks = sorted(set(pretrain_tasks + target_tasks), key=lambda x: x.name)
    model = build_model(clargs, vocab, word_embs, tasks)
    for i in range(0, 100, args.eval_every)
       if os.path.exists(os.path.join(clargs.run_dir, "model_state_main_epoch_" + str())) == False:
            continue
        else:
            print("Epoch " + str(i) + ":\n")
        macro_best = glob.glob(os.path.join(clargs.run_dir,
                                            "model_state_main_epoch*"))
        load_model_state(model, macro_best[-1], args.cuda)
        corpus = data.Corpus(vocab._token_to_index['tokens'])
        f1_list = [[], [], []]
        prec_list = []
        reca_list = []
        model.eval()
        for i in range(len(corpus.test)):
            st = corpus.test[i].reshape(1, -1).cuda()
            ta = torch.cat([corpus.test[i][1:].cuda(), corpus.test[i][:1].cuda()]).reshape(1, -1)
            inp = {}
            tmp = {}
            tmp['words'] = st
            inp['input'] = tmp
            tmp1 = {}
            tmp1['words'] = ta
            inp['targs'] = tmp1
            inp['targs_b'] = tmp1
            #sent_encoder(batch['input'], task)
            _, _ = model.sent_encoder.forward(tmp, tasks[0])

            distances = model.sent_encoder._phrase_layer.distances
            for layerID in [0, 1, 2]:
                dc = distances[layerID][1:-1]
                sen_cut = corpus.test[i][1:-1]
                sen_tree = corpus.test_trees[i]
                parse_tree = build_tree(dc.cpu().detach(), sen_cut)
                model_out, _ = get_brackets(parse_tree)
                std_out, _ = get_brackets(sen_tree)
                overlap = model_out.intersection(std_out)
                prec = float(len(overlap)) / (len(model_out) + 1e-8)
                reca = float(len(overlap)) / (len(std_out) + 1e-8)
                if len(std_out) == 0:
                    reca = 1.
                    if len(model_out) == 0:
                        prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)
                f1_list[layerID].append(f1)

        print("\n")
        for layerId in [0, 1, 2]:
            print("Layer " + str(layerId))
            print(mean(f1_list[layerId]))
            print("\n")
