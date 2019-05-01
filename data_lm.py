import os
import re
import pickle
import copy

import numpy
import torch
import nltk

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']



class Corpus(object):
    def __init__(self, dictionary):

        self.dictionary=dictionary
        file_path="/Users/anhadmohananey/Downloads/stanford-corenlp-full-2018-02-27/toronto_dev.ps"
        self.test, self.test_sens, self.test_trees, self.test_nltktrees = self.tokenize(file_path)

    def tokenize(self, fileid):
        filen=open(fileid, "r")
        lowercase=True
        sens_idx = []
        sens = []
        trees = []
        nltk_trees = []
        for line in filen.readlines():
            transitions=[]
            tokens=[]
            for word in line.split(' '):
                if word[0] != "(":
                    if word.strip() == ")":
                        transitions.append(1)
                    else:
                        # Downcase all words to match GloVe.
                        if lowercase:
                            tokens.append(word.lower())
                        else:
                            tokens.append(word)
                        transitions.append(0)

            arr=[]
            tmp=[]
            stack=[]
            words = ['<EOS>'] + tokens + ['<EOS>']
            sens.append(words)
            tokens=tokens[::-1]
            for x in transitions:
                if x == 0:
                    #shift
                    stack.append(tokens.pop())
                else:
                    a1=stack.pop()
                    a2=stack.pop()
                    stack.append([a2,a1])
            idx = []
            for word in words:
                if word not in self.dictionary:
                    idx.append(self.dictionary["@@UNKNOWN@@"])
                else:
                    idx.append(self.dictionary[word])
            sens_idx.append(torch.LongTensor(idx))
            trees.append(stack[0])
            nltk_trees.append(stack[0])

        return sens_idx, sens, trees, nltk_trees
