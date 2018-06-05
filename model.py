
from utils import *
from preprocess import *
from beam_search import BeamIterator
from encoder import Encoder
from decoder import Decoder
from memory import Memory

import torch.nn as nn
from torch.autograd import Variable

import pickle
import tarfile

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Seq2Seq(nn.Module):
    # Either pass in the arguments to initialize the model, or leave them blank to create an empty model
    # Initializing the model requires a filled dictionary, width of hidden layers, and number of layers for each RNN
    def __init__(self):
        super(Seq2Seq,self).__init__()
        self.base_rnn = None
        self.wd = None
        self.encoder = None
        self.decoder = None
        self.memory = None

    def init_model(self, wd, hidden_size, e_layers, d_layers, mem_size):
        self.wd = wd
        self.encoder = Encoder(wd.n_words, hidden_size, n_layers=e_layers, base_rnn=nn.GRU)
        self.decoder = Decoder(hidden_size, wd.n_words, n_layers=d_layers, base_rnn=nn.GRU)
        self.memory = Memory(wd, self.encoder, mem_size)
        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.memory = self.memory.cuda()
        return self

    def _apply_batched(self, batch, tf_ratio=0.5, train=False, memory=False):
        input_batch, input_lengths, target_batch, target_lengths, indices = batch
        batch_size = input_batch.size(1)

        # Pass the input batch through the encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batch, input_lengths, None)

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_INDEX] * batch_size))
        latent_vector = encoder_hidden[:self.decoder.n_layers]  # Use last (forward) hidden state from encoder

        if memory:
            decoder_hidden = self.memory(latent_vector, indices=indices)
        else:
            decoder_hidden = latent_vector

        max_target_length = max(target_lengths)
        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.decoder.output_size))

        # Move new Variables to CUDA
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        # Pass the batch iteratively through the decoder
        for t in range(max_target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, lengths=input_lengths)
            all_decoder_outputs[t] = decoder_output

            r = np.random.rand()
            if train and r < tf_ratio:
                decoder_input = target_batch[t]  # Next input is current target
            else:
                _, topi = decoder_output.data.topk(1)
                decoder_input = Variable(topi[0, :, 0])

        return all_decoder_outputs

    def _apply_single(self, input_seq, memory=False, mask_index=None):
        input_lengths = [len(input_seq)]
        input_seqs = [input_seq]
        # with torch.no_grad():
        input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

        if USE_CUDA:
            input_batches = input_batches.cuda()

        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)

        # Run through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        latent_vector = encoder_hidden[:self.decoder.n_layers]  # Use last (forward) hidden state from encoder

        if memory:
            decoder_hidden = self.memory(latent_vector, indices=mask_index)
        else:
            decoder_hidden = latent_vector

        # Initialize and execute the beam search
        iterator = BeamIterator(self, decoder_hidden, 3, 3)
        decoded_words = iterator.search()

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        return decoded_words

    # Trains on a single batch
    def train_naive(self, batch, tf_ratio=0.5):
        input_batch, input_lengths, target_batch, target_lengths = batch

        all_decoder_outputs = self._apply_batched(batch, tf_ratio=tf_ratio, train=True, memory=False)

        # Calculate and backpropagate the loss
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batch.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss.backward()

        return loss.data[0]

    def train_memory(self, batch, tf_ratio=0.5):
        input_batch, input_lengths, target_batch, target_lengths, indices = batch

        all_decoder_outputs = self._apply_batched(batch, tf_ratio=tf_ratio, train=True, memory=True)

        # Calculate and backpropagate the loss
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batch.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss.backward()

        return loss.data[0]


    def validate(self, batch):
        input_batch, input_lengths, target_batch, target_lengths = batch

        all_decoder_outputs = self._apply_batched(batch)

        # Calculate and backpropagate the loss
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batch.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )

        return loss.data[0]

    def forward(self, inputs, batched=False):
        if batched:
            return self._apply_batched(inputs)
        else:
            return self._apply_single(inputs)

    def _old_score(self, pair):
        out = self._apply_single(pair[0])
        out_line = self.wd.to_words(out)[1:-1]
        out_true = self.wd.to_words(pair[1])[:-1]

        smooth=SmoothingFunction().method1

        score_bleu1 = sentence_bleu([out_true], out_line, weights=(1.0, 0.0, 0.0, 0.0),
                                    smoothing_function=smooth)
        score_bleu2 = sentence_bleu([out_true], out_line, weights=(0.5, 0.5, 0.0, 0.0),
                                    smoothing_function=smooth)
        score_bleu3 = sentence_bleu([out_true], out_line, weights=(0.33, 0.33, 0.33, 0.0),
                                    smoothing_function=smooth)
        score_bleu4 = sentence_bleu([out_true], out_line, weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=smooth)
        return torch.FloatTensor([score_bleu1, score_bleu2, score_bleu3, score_bleu4])

    def _new_score(self, pair):
        out = self._apply_single(pair[0])
        out_line = remove_punctuation(self.wd.to_words(out)[1:-1])
        out_true = remove_punctuation(self.wd.to_words(pair[1])[:-1])

        smooth = SmoothingFunction(epsilon=1e-8).method1

        score_bleu1 = sentence_bleu([out_true], out_line, weights=(1.0, 0.0, 0.0, 0.0),
                                    smoothing_function=smooth)
        score_bleu2 = sentence_bleu([out_true], out_line, weights=(0.5, 0.5, 0.0, 0.0),
                                    smoothing_function=smooth)
        score_bleu3 = sentence_bleu([out_true], out_line, weights=(0.33, 0.33, 0.33, 0.0),
                                    smoothing_function=smooth)
        score_bleu4 = sentence_bleu([out_true], out_line, weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=smooth)
        return torch.FloatTensor([score_bleu1, score_bleu2, score_bleu3, score_bleu4])

    def score_sets(self, sets):
        old_scores = torch.zeros(4)
        new_scores = torch.zeros(4)
        total_pairs=0
        for set in sets:
            n_pairs = len(set)
            total_pairs += n_pairs
            self.memory.reset_memory(n_pairs)
            for pair in set:
                old_scores += self._old_score(pair)
                new_scores += self._new_score(pair)
                self.memory.add_pair(pair)
        old_scores *= 100.0 / total_pairs
        new_scores *= 100.0 / total_pairs

        return old_scores, new_scores

    def export_state(self, dir, label):
        print("Saving models.")

        cwd = os.getcwd() + '/'

        enc_out = dir + ENC_FILE
        dec_out = dir + DEC_FILE
        mem_out = dir + MEM_FILE
        inf_out = dir + INF_FILE

        torch.save(self.encoder.state_dict(), enc_out)
        torch.save(self.decoder.state_dict(), dec_out)
        torch.save(self.memory.state_dict(), mem_out)

        info = open(inf_out, 'w')
        info.write(str(self.encoder.hidden_size) + "\n" + str(self.encoder.n_layers) + "\n" + str(
            self.decoder.n_layers))
        info.close()

        files = [enc_out, dec_out, mem_out, inf_out]

        print("Bundling models")

        tf = tarfile.open(cwd + dir + label, mode='w')
        for file in files:
            tf.add(file)
        tf.close()

        for file in files:
            os.remove(file)

        print("Finished saving models.")

    def import_state(self, model_file, active_dir=TEMP_DIR, encoder=True, decoder=True, memory=True):
        print("Loading models.")
        cwd = os.getcwd() + '/'
        model_dir = os.path.abspath(os.path.join(model_file, os.pardir))

        tf = tarfile.open(model_file)
        # extract directly to current model directory
        for member in tf.getmembers():
            if member.isreg():
                member.name = os.path.basename(member.name)
                tf.extract(member, path=active_dir)
        tf.close()

        info = open(active_dir + INF_FILE, 'r')
        lns = info.readlines()
        hidden_size, e_layers, d_layers, using_lstm = [int(i) for i in lns[:5]]

        self.wd = WordDict().import_dicts(model_dir+DICTS_FILE)

        self.encoder = Encoder(self.wd.n_words, hidden_size, n_layers=e_layers, base_rnn=nn.GRU)
        self.decoder = Decoder(hidden_size, self.wd.n_words, n_layers=d_layers, base_rnn=nn.GRU)

        if not USE_CUDA:
            if encoder:
                self.encoder.load_state_dict(torch.load(cwd + TEMP_DIR + ENC_FILE, map_location=lambda storage, loc: storage))
            if decoder:
                self.decoder.load_state_dict(torch.load(cwd + TEMP_DIR + DEC_FILE, map_location=lambda storage, loc: storage))
        else:
            if encoder:
                self.encoder.load_state_dict(torch.load(cwd + TEMP_DIR + ENC_FILE))
            if decoder:
                self.decoder.load_state_dict(torch.load(cwd + TEMP_DIR + DEC_FILE))
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.memory = Memory(self.wd, self.encoder, mem_size=1000)
        if memory:
            if not USE_CUDA:
                self.memory.load_state_dict(torch.load(cwd + TEMP_DIR + MEM_FILE, map_location=lambda storage, loc: storage))
            else:
                self.memory.load_state_dict(torch.load(cwd + TEMP_DIR + MEM_FILE))
                self.memory = self.memory.cuda()
        else:
            if USE_CUDA:
                self.memory=self.memory.cuda()

        self.encoder.eval()
        self.decoder.eval()
        self.memory.eval()

        self.memory.reset_memory()

        print("Loaded models.")

        return self