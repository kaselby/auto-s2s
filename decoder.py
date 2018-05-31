from utils import *

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, base_rnn=nn.GRU):
        super(Decoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.base_rnn = base_rnn

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = self.base_rnn(hidden_size, hidden_size, n_layers)
        self.rnn_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, last_hidden, lengths=None):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        rnn_output, hidden = self.rnn(embedded, last_hidden)

        # Finally predict next token
        output = self.rnn_out(rnn_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden

