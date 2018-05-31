from utils import *

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, base_rnn=nn.GRU):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.base_rnn = base_rnn

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = self.base_rnn(hidden_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)

        embedded = self.embedding(input_seqs)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)

        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden
