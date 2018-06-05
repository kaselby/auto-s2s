from preprocess import *

import os

import math
import datetime
import time

import torch.nn as nn
import torch.nn.functional as F


#
#   Misc
#

def is_nan(var):
    isnan = (var != var)
    if type(isnan) != bool:
        isnan = isnan.any()
    return isnan

def invert_permutation(p):
    N=len(p)
    x=list(range(N))
    for i in range(N):
        x[p[i]]=i
    return x


#
#   Masking
#

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    if USE_CUDA:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()

    loss = losses.sum() / length.float().sum()
    return loss




#
#   Saving and Loading Models
#

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def init_save(args, val_indices, model_dir=MODEL_DIR):
    t = datetime.datetime.now()
    timestamp = str(t.day) + "_" + str(t.hour) + "_" + str(t.minute)
    path = model_dir + "s2s_" + timestamp + "/"
    if not os.path.isdir(path):
        os.mkdir(path)

    if args is not None:
        log_file = open(path + LOG_FILE, 'a')
        arg_dict = vars(args)
        log_file.write("Training Parameters:\n")
        for (arg, val) in arg_dict.items():
            log_file.write(str(arg) + ": " + str(val) + '\n')
        log_file.write("\n")
        log_file.close()
        print("Model parameters saved.")

    if val_indices is not None:
        export_val(val_indices, path)
        print("Validation set saved.")

    return path

def save_logs(logs, path):
    log_file = open(path + LOG_FILE, 'a')
    try:
        for log in logs:
            log_file.write(str(log))
    except TypeError:
        log_file.write(str(logs))
    log_file.write("\n")
    log_file.close()

def save_scores(old_scores, new_scores, path, old_file="scores_old.dat", new_file="scores_new.dat"):
    old_score_file = open(path + old_file, 'a')
    new_score_file = open(path + new_file, 'a')

    old_score_file.write(str(old_scores.numpy()))
    new_score_file.write(str(new_scores.numpy()))

    old_score_file.write("\n")
    new_score_file.write("\n")
    new_score_file.close()
    old_score_file.close()

def export_val(val_indices, path):
    val_file = open(path + VAL_FILE, 'w')
    for index in val_indices:
        val_file.write(str(index)+",")
    val_file.close()

def import_val(path):
    val_file = open(path + VAL_FILE, 'w')
    raw_indices = val_file.readline()
    val_file.close()
    return [int(i) for i in raw_indices]