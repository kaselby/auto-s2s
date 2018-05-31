import torch
from torch.autograd import Variable

import numpy as np

#
#   Tokenizing constants
#

SOS_INDEX = 0
EOS_INDEX = 1
UNK_INDEX = 2
BRK_INDEX = 3
PAD_INDEX = 4
ELP_INDEX = 5
NL_INDEX = 6

UNK = "<unk>"
BRK = "<brk>"
EOS = "<eos>"
SOS = "<sos>"
PAD = "<pad>"
ELP = "<elp>"
NL = "<nl>"
RST = "<rst>"
END = "<end>"
PAIR = "<p>"

PUNCTUATION=",.?!;:"
TOKENS = {r'\.\.\.': ' '+ELP+' '}
RESERVED_I2W = {SOS_INDEX: SOS, EOS_INDEX: EOS, UNK_INDEX: UNK, BRK_INDEX: BRK,
            PAD_INDEX: PAD, ELP_INDEX: ELP, NL_INDEX: NL}
RESERVED_W2I = dict((v,k) for k,v in RESERVED_I2W.items())
RMV_TOKENS = [EOS, SOS]

#
#   I/O Constants
#

TEMP_DIR = "current_model/"
DATA_DIR = "data/"
MODEL_DIR = "trained_models/"
ENC_FILE = "enc.pt"
DEC_FILE = "dec.pt"
MEM_FILE = "mem.pt"
I2W_FILE = "i2w.dict"
W2I_FILE = "w2i.dict"
U_I2W_FILE = "unk_i2w.dict"
U_W2I_FILE = "unk_w2i.dict"
INF_FILE = "info.dat"
TRAIN_INF_FILE = "train_info.dat"
FIG_FILE = "losses.png"
LOSS_FILE = "losses.dat"
SCORE_FILE = "scores.dat"
LOG_FILE = "log.txt"
DATA_FILE = "conv_cornell.txt"

DICTS_FILE = "vocab.tar"

TRAIN_FILE = "train.dat"
VAL_FILE = "val.dat"
TEST_FILE = "test.dat"


#
#   Model Constants
#

USE_CUDA = torch.cuda.is_available()

MAX_SENTENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH = MAX_SENTENCE_LENGTH + 2