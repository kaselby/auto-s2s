from minibatches import *

from torch.autograd import Variable


import tarfile
import pickle
import os

import re
import unicodedata
import random
import math

import numpy as np

#
#   Dictionary class for storing dictionaries from words to indices and vice versa
#

class WordDict(object):
    def __init__(self, dicts=None):
        if dicts == None:
            self._init_dicts()
        else:
            self.word2index, self.index2word, self.word2count, self.n_words = dicts

    def _init_dicts(self):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.index2word.update(RESERVED_I2W)
        self.word2index.update(RESERVED_W2I)

        self.n_words = len(RESERVED_I2W)  # number of words in the dictionary

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if not word in RESERVED_W2I:
            if not word in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def remove_unknowns(self, cutoff):
        # find unknown words
        unks = []
        for word, count in self.word2count.items():
            if count <= cutoff and word not in RESERVED_W2I:
                unks.append(word)

        # remove unknown words
        for word in unks:
            del self.index2word[self.word2index[word]]
            del self.word2index[word]
            del self.word2count[word]

        # reformat dictionaries so keys get shifted to correspond to removed words
        old_w2i = self.word2index
        self._init_dicts()
        for word, index in old_w2i.items():
            if word not in RESERVED_W2I:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
        self.n_words = self.n_words

        return unks

    def to_indices(self, words):
        indices = []
        for word in words:
            if word in self.word2index:
                indices.append(self.word2index[word])
            else:
                indices.append(self.word2index[UNK])
        return indices

    def to_words(self, indices):
        words = []
        for index in indices:
            if index in self.index2word:
                words.append(self.index2word[index])
            else:
                words.append(UNK)
        return words

    def export_dicts(self, path, label):
        cwd = os.getcwd()

        i2w_out = path + I2W_FILE
        w2i_out = path + W2I_FILE

        i2w = open(i2w_out, 'wb')
        pickle.dump(self.index2word, i2w)
        i2w.close()
        w2i = open(w2i_out, 'wb')
        pickle.dump(self.word2index, w2i)
        w2i.close()

        files = [i2w_out, w2i_out]

        tf = tarfile.open(cwd + path + label, mode='w')
        for file in files:
            tf.add(file)
        tf.close()

        for file in files:
            os.remove(file)

    def import_dicts(self, path, active_dir=TEMP_DIR):
        cwd = os.getcwd()
        tf = tarfile.open(path)

        # extract directly to current model directory
        for member in tf.getmembers():
            if member.isreg():
                member.name = os.path.basename(member.name)
                tf.extract(member, path=active_dir)

        i2w = open(cwd + TEMP_DIR + I2W_FILE, 'rb')
        w2i = open(cwd + TEMP_DIR + W2I_FILE, 'rb')

        i2w_dict = pickle.load(i2w)
        w2i_dict = pickle.load(w2i)
        w2i.close()
        i2w.close()

        self.index2word = i2w_dict
        self.word2index = w2i_dict
        self.n_words = len(self.index2word)

#
#   4-stage string preprocessing:
#

#
#   Stage 1: Remove unneeded tokens and punctuation
#

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize(s):
    s = unicode_to_ascii(s.lower().strip())
    for token, flag in TOKENS.items():
        s = re.sub(token, flag, s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?<>']+", r" ", s)
    return s


#
#   Stage 2: Truncate and separate into word list
#

def separate(s, max_len=MAX_SENTENCE_LENGTH, separator=" ", rmv=False):
    sep_lines = s.split(separator)
    if rmv:
        return sep_lines if len(sep_lines) <= max_len else BRK
    else:
        return s.split(separator)[:max_len]


#
#   Stage 3a: Construct message/response pairs and separate into training and validation sets.
#

def get_pairs(lines):
    pairs = []
    msg = None
    resp = None
    addpair = False
   # print("Collecting pairs of index lists.")
    for i in range(len(lines)):
        if not BRK in lines[i]:
            resp = lines[i]
            if addpair == True:
                pairs.append([msg, resp])
            msg = resp
            addpair = True
        else:
            addpair = False
    n_pairs = len(pairs)
   # print(str(n_pairs) + " pairs of index lists collected.")
    return pairs

def get_validation_set(movies, val_frac):
    if val_frac <= 0:
        return movies, None, None
    n = len(movies)
    n_val = math.floor(n*val_frac)
    print(n,n_val)
    indices = random.sample(range(n), n_val)
    train_sets = []
    val_sets = []
    for i in range(n):
        if i not in indices:
            train_sets.append(movies[i])
        else:
            val_sets.append(movies[i])
    val_pairs = sum([len(v) for v in val_sets])
    train_pairs = sum([len(t) for t in train_sets])
    print("Training and validation sets generated.")
    print(train_pairs, "training pairs total.", val_pairs, "validation pairs total.")
    return train_sets, val_sets, indices

def get_val_from_indices(movies, val_indices):
    train_set = []
    val_set = []
    for i in range(len(movies)):
        if i in val_indices:
            val_set.append(movies[i])
        else:
            train_set.append(movies[i])
    return train_set, val_set

#
#   Stage 3b: Insert SOS/EOS tokens and convert to indices
#

def tokenize(s, wd):
    return wd.to_indices(s) + [EOS_INDEX]

def tokenize_lines(lines, wd):
    return [tokenize(l, wd) for l in lines]

def tokenize_pairs(pairs, wd):
    tokenized_pairs = []
    for pair in pairs:
        tokenized_pairs.append([tokenize(s, wd) for s in pair])
    return tokenized_pairs

#
#   Stage 4: Construct batches and convert to Pytorch variable tensors
#

#   see minibatches.py

#
#   Full preprocessing pipeline on an input query:
#

def parse_query(msg, wd):
    return tokenize(separate(normalize(msg)), wd)



#
#   Clean model outputs
#

def clean_resp(raw_resp, rmv_tokens=list(RESERVED_W2I.keys())):
    resp = [w for w in raw_resp if not w in rmv_tokens]
    return " ".join(resp)

def remove_punctuation(sequence):
    seq_out = []
    for s in sequence:
        if not s in PUNCTUATION:
            seq_out.append(s)
    return seq_out

#
#   Loading input from files.
#

def import_csv(datafile, max_lines=-1, unk_thresh=5, new_dict=True):
    if new_dict:
        wd = WordDict()
    else:
        wd = None
    print("Reading input...")
    sets = [[]]
    with open(datafile, 'r') as infile:
        count = 0
        for line in infile:
            if max_lines > 0 and count >= max_lines:
                break
            if RST in line:
                sets.append([])
                continue
            split_line = separate(normalize(line))
            if new_dict:
                wd.add_sentence(split_line)
            sets[-1].append(split_line)
            count += 1
    print("Input read.")
    print(str(sum([len(s) for s in sets])), "total lines.")

    if new_dict:
        print(str(wd.n_words), "total unique words.")
        unks = wd.remove_unknowns(unk_thresh)
        print(str(len(unks)), "words removed.", str(wd.n_words), "words remaining in vocabulary.")

    if new_dict:
        return sets, wd
    else:
        return sets

def list_to_string(msg):
    return ",".join(msg)

def export_pairs(pairs, path):
    outfile = open(path, 'w')
    for pair in pairs:
        outfile.write("--\n"+list_to_string(pair[0])+"\n"+list_to_string(pair[1])+"\n")
    outfile.close()

def import_pairs(path):
    infile = open(path, 'r')
    pairs=[]
    done=False
    while not done:
        delim = infile.readline()
        if delim != "":
            msg = separate(normalize(infile.readline()))
            resp = separate(normalize(infile.readline()))
            pairs.append((msg, resp))
        else:
            done=True
    return pairs

#
#   Stuff
#

def partition_movies(all_movies, set_size, discard_excess=False):
    N_movies = len(all_movies)
    if discard_excess:
        N_sets = math.floor(N_movies/set_size)
    else:
        N_sets = math.ceil(N_movies/set_size)
    p = np.random.permutation(N_movies)
    sets=[]
    index=0
    for i in range(N_sets):
        min_index = i * set_size
        max_index = min(N_movies, min_index + set_size)
        movies = [all_movies[k] for k in p[min_index:max_index]]
        length = 0
        for movie in movies:
            movies.append(movie)
            length += len(movie)
        sets.append(movies, index, index+length)
        index += length
    return sets

def process_movies(movies, val_frac=0.1):
    N_movies = len(movies)
    N_val = int(np.floor(N_movies * val_frac))
    N_train = N_movies - N_val

    p=np.random.permutation(N_movies)
    val_indices = p[:N_val] if N_val > 0 else None
    train_indices = p[N_val:] if N_val > 0 else p

    val_set = [movies[k] for k in val_indices] if N_val > 0 else None
    train_set = [movies[k] for k in train_indices]

    return train_set, val_set, val_indices

def process_lines(lines, wd):
    convs = []
    for i in range(len(lines)):
        if not BRK in lines[i]:
            convs[-1].append(tokenize(lines[i], wd))
        else:
            convs.append([])
    return convs

def convs_to_pairs(convs):
    pairs = []
    for conv in convs:
        for i in range(len(conv)-1):
            pairs.append(conv[i], conv[i+1])