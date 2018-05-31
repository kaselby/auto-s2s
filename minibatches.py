from constants import *

def pad_seq(seq, max_length):
    seq += [PAD_INDEX for i in range(max_length - len(seq))]
    return seq

def batch_inputs(messages, sort=True, var=True):
    N=len(messages)
    if sort:
        # Zip into pairs, sort by length (descending), unzip
        sorted_seqs = sorted(zip(messages, list(range(N))), key=lambda p: len(p[0]), reverse=True)
        seqs, p = zip(*sorted_seqs)
    else:
        seqs = messages

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_padded).transpose(0, 1)

    if var:
        input_var = Variable(input_var)

    if USE_CUDA:
        input_var = input_var.cuda()

    if sort:
        batch = (input_var, input_lengths, p)
    else:
        batch = (input_var, input_lengths)

    return batch

def batch_pairs(inputs, targets, sort=True):
    if sort:
        # Zip into pairs, sort by length (descending), unzip
        seq_pairs = sorted(zip(inputs, targets), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)
    else:
        input_seqs, target_seqs = inputs, targets

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return (input_var, input_lengths, target_var, target_lengths)

def random_batches_auto(batch_size, lines):
    n_batches = int(len(lines) / batch_size)
    p = np.random.permutation(len(lines))

    batches = []
    for i in range(n_batches):
        input_seqs = []
        target_seqs = []
        # Choose random pairs
        for j in range(batch_size):
            line = lines[p[i*batch_size+j]]
            input_seqs.append(line)
            target_seqs.append(line)

        batches.append(batch_pairs(input_seqs, target_seqs))

    return batches

def batches_by_length(batch_size, pairs):
    n_pairs = len(pairs)
    n_batches = int(n_pairs / batch_size)

    buckets = bucketize(pairs, MAX_SEQUENCE_LENGTH+1, key=lambda p: len(p[0]))

    batches = []
    done_all=False
    current_bucket = MAX_SEQUENCE_LENGTH
    current_index=0
    while not done_all:
        input_seqs = []
        target_seqs = []

        n_batch = batch_size

        done_batch=False
        while not done_batch:
            bucket = buckets[current_bucket]
            n_bucket = len(bucket)
            n_left = n_bucket - current_index
            n_current = min(n_batch, n_left)

            for i in range(current_index, current_index + n_current):
                input_seq, target_seq = bucket[i]
                input_seqs.append(input_seq)
                target_seqs.append(target_seq)

            current_index += n_current
            n_batch -= n_current

            if n_batch > 0:
                if current_bucket > 0:
                    current_bucket -= 1
                    current_index = 0
                else:
                    done_all=True
                    done_batch=True
            else:
                batch = batch_pairs(input_seqs, target_seqs)
                batches.append(batch)
                done_batch=True

    return batches


def bucketize(raw_list, max_len, key=len, shuffled=True):
    sorted_list = [[] for i in range(max_len)]
    for i in range(len(raw_list)):
        element = raw_list[i]
        val = key(element)
        sorted_list[val].append(element)

    if shuffled:
        for i in range(max_len):
            pairs_i = sorted_list[i]
            new_pairs = []
            n = len(pairs_i)
            p = np.random.permutation(n)
            for j in range(n):
                k = p[j]
                new_pairs.append(pairs_i[k])
            sorted_list[i] = new_pairs

    return sorted_list


#
#   Memory batching
#

def memory_batch_pairs(inputs, targets, seq_indices, sort=True):
    if sort:
        # Zip into pairs, sort by length (descending), unzip
        seq_pairs = sorted(zip(inputs, targets, seq_indices), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs, seq_indices = zip(*seq_pairs)
    else:
        input_seqs, target_seqs, seq_indices = inputs, targets, seq_indices

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    index_var = Variable(torch.LongTensor(seq_indices))

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return (input_var, input_lengths, target_var, target_lengths, seq_indices)


def memory_random_batches(batch_size, movies):
    cumulative_lengths = [0]
    tot=0
    for movie in movies:
        tot += len(movie)
        cumulative_lengths.append(tot)
    n_pairs = sum(cumulative_lengths)
    n_batches = int(n_pairs / batch_size)
    p = np.random.permutation(n_pairs)

    batches = []
    for i in range(n_batches):
        input_seqs = []
        target_seqs = []
        seq_indices = []
        # Choose random pairs
        for j in range(batch_size):
            index = int(p[i*batch_size+j])
            movie_index, pair_index = to_set(index, cumulative_lengths)
            pair = movies[movie_index][pair_index]
            input_seqs.append(pair[0])
            target_seqs.append(pair[1])
            seq_indices.append(pair_index)

        batch = memory_batch_pairs(input_seqs, target_seqs, seq_indices)
        batches.append(batch)
    return batches


def to_set(i, cumulative_lengths):
    for j in range(len(cumulative_lengths)-1):
        if i < cumulative_lengths[j+1]:
            return (j, i-cumulative_lengths[j])

def to_index(i, j, cumulative_lengths):
    return cumulative_lengths[i]+j