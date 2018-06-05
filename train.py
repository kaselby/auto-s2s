from utils import *
from training_handler import *

import argparse

def create_model(args):
    # Preprocess data
    sets, wd = import_csv(DATA_DIR + args.datafile, max_lines=args.max_lines, unk_thresh=args.unk_thresh)
    pair_sets = [get_pairs(s) for s in sets]

    tokenized_sets = [tokenize_pairs(ps, wd) for ps in pair_sets]

    n_pairs = [len(s) for s in tokenized_sets]
    max_size=sum(n_pairs)

    train_sets, val_sets, val_indices = get_validation_set(tokenized_sets, val_frac=args.val_frac)
    model_path = init_save(args, val_indices)

    print("Variables processed.")

    model = Seq2Seq().init_model(wd, args.hidden_size, args.layers, args.layers, max_size)
    del sets, pair_sets, tokenized_sets
    return model, model_path, train_sets, val_sets

def load_model(path):
    model = Seq2Seq().import_state(memory=False)

    sets = import_csv(DATA_DIR + args.datafile, max_lines=args.max_lines, unk_thresh=args.unk_thresh, new_dict=False)
    pair_sets = [get_pairs(s) for s in sets]

    tokenized_sets = [tokenize_pairs(ps, model.wd) for ps in pair_sets]

    n_pairs = [len(s) for s in tokenized_sets]
    max_size=sum(n_pairs)

    val_indices = import_val(path)
    train_sets, val_sets = get_val_from_indices(tokenized_sets, val_indices)

    model.memory.reset_memory(max_size)

    return model, train_sets, val_sets

def init_parser():
    parser = argparse.ArgumentParser(description='Sequence to sequence chatbot model.')
    parser.add_argument('-df', dest='datafile', action='store', type=str, default=DATA_FILE)
    parser.add_argument('-v', dest='val_frac', action='store', type=float, default=0.1)
    parser.add_argument('-max', dest='max_lines', action='store', type=int, default=-1)
    parser.add_argument('-e', dest='epochs', action='store', type=int, default=60)
    parser.add_argument('-hs', dest='hidden_size', action='store', type=int, default=1024)
    parser.add_argument('-bs', dest='batch_size', action='store', type=int, default=256)
    parser.add_argument('-lr', dest='learning_rate', action='store', type=float, default=0.0005)
    parser.add_argument('-l', dest='layers', action='store',type=int, default=1)
    parser.add_argument('-u', dest='unk_thresh', action='store', type=int, default=1)
    parser.add_argument('-s', dest='save_interval', action='store', type=int, default=5)
    parser.add_argument('-eps', dest='eps', action='store', type=float, default=1e-3)
    parser.add_argument('-t', dest='trained_model', action='store', type=str, default='')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Import arguments passed from argparser and cast them to appropriate types
    args = init_parser()

    if args.trained_model is not '':
        model_dir = args.trained_model
        model, train_sets, val_sets = load_model(model_dir)
        trainer = TrainingHandler(model, train_sets, val_sets, args.learning_rate, model_dir, eps=args.eps)
        trainer.train_memory(args.epochs, args.batch_size)
    else:
        model, model_dir, train_sets, val_sets = create_model(args)
        trainer = TrainingHandler(model, train_sets, val_sets, args.learning_rate, model_dir, eps=args.eps)
        trainer.train_autoencoder(args.epochs, args.batch_size, save_interval=args.save_interval)