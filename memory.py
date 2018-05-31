from encoder import *
from preprocess import *

import copy


class Memory(nn.Module):
    def __init__(self, wd, encoder, mem_size, temperature=1):
        super(Memory, self).__init__()
        self.wd = wd
        self.encoder = None
        self.hidden_size = encoder.hidden_size
        self.memory_size = mem_size

        self.raw_messages = None
        self.raw_responses = None
        self.encoded_messages = None
        self.encoded_responses = None
        self.index_map = None

        self.temperature = temperature

        self.memory_used = 0
        self.raw_memory_used = 0

        self.transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # come back to bias term - bias irrelevant due to softmax right?

        self.init_eye()

        self.reset_memory()
        self.update_encoder(encoder)

    def init_eye(self):
        nn.init.eye(self.transform.weight)

    def init_random(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        nn.init.uniform(self.transform.weight, -stdv, stdv)

    def expand_memory_pad(self, new_size):
        assert new_size > self.memory_size
        new_messages = torch.zeros(self.hidden_size, new_size)
        new_responses = torch.zeros(self.hidden_size, new_size)
        new_index_map = [-1]*new_size

        if self.memory_used > 0:
            new_messages[:,:self.memory_used] = self.encoded_messages[:,:self.memory_used]
            new_responses[:, :self.memory_used] = self.encoded_responses[:, :self.memory_used]
            new_index_map[:self.memory_used] = self.index_map[:self.memory_used]

        if USE_CUDA:
            new_messages = new_messages.cuda()
            new_responses = new_responses.cuda()

        self.encoded_messages=new_messages
        self.encoded_responses=new_responses
        self.index_map=new_index_map
        self.memory_size = new_size

    def reset_memory(self, mem_size=-1):
        if mem_size > 0:
            self.memory_size = mem_size
        self.raw_messages = []
        self.raw_responses = []
        self.encoded_messages = torch.zeros(self.hidden_size, self.memory_size)
        self.encoded_responses = torch.zeros(self.hidden_size, self.memory_size)
        self.index_map = [-1]*self.memory_size
        self.memory_used = 0
        self.raw_memory_used = 0

        if USE_CUDA:
            self.encoded_messages = self.encoded_messages.cuda()
            self.encoded_responses = self.encoded_responses.cuda()

    def _encode_message(self, message):
        input_lengths = [len(message)]
        input_seqs = [message]
        input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)

        if USE_CUDA:
            input_batches = input_batches.cuda()

        # Set to not-training mode to disable dropout
        self.encoder.train(False)

        # Run through encoder
        encoder_outputs, encoder_hidden = self.encoder(
            input_batches, input_lengths, None)

        return encoder_hidden[0, :, :]  #forward vs backward hidden state etc...

    def _encode_batch(self, input_batch, input_lengths):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)

        # Run through encoder
        encoder_outputs, encoder_hidden = self.encoder(
            input_batch, input_lengths, None)

        return encoder_hidden[0, :, :]

    def _encode_pairs(self, messages, responses):
        message_batch, message_lengths, message_p = batch_inputs(messages)
        response_batch, response_lengths, response_p = batch_inputs(responses)

        encoded_messages = self._encode_batch(message_batch, message_lengths)
        encoded_responses = self._encode_batch(response_batch, response_lengths)

        message_revert = invert_permutation(message_p)
        response_revert = invert_permutation(response_p)
        encoded_messages = encoded_messages[message_revert, :]
        encoded_responses = encoded_responses[response_revert, :]

        return encoded_messages, encoded_responses

    def add_pair(self, pair, normalize=True):
        message, response = pair
        self.raw_messages.append(message)
        self.raw_responses.append(response)

        encoded_message = self._encode_message(message).data.view(-1, 1)
        encoded_response = self._encode_message(response).data.view(-1, 1)

        if normalize:
            #normalize messages but not responses
            encoded_message /= encoded_message.norm()
            encoded_response /= encoded_response.norm()

        if self.memory_used < self.memory_size:
            self.encoded_messages[:, self.memory_used] = encoded_message
            self.encoded_responses[:, self.memory_used] = encoded_response
            self.index_map[self.memory_used] = self.raw_memory_used
            self.memory_used += 1
        else:
            print("Cap reached.")
            self.encoded_messages = torch.cat((self.encoded_messages[:, 1:], encoded_message), 1)
            self.encoded_responses = torch.cat((self.encoded_responses[:, 1:], encoded_response), 1)
            self.index_map = self.index_map[1:] + [self.raw_memory_used]

        self.raw_memory_used += 1

    def add_pairs(self, pairs, normalize=True):
        messages=[]
        responses=[]
        for pair in pairs:
            message, response = pair
            messages.append(message)
            responses.append(response)
            self.raw_messages.append(message)
            self.raw_responses.append(response)

        encoded_messages, encoded_responses = self._encode_pairs(messages, responses)
        encoded_messages = encoded_messages.data.transpose(0,1)
        encoded_responses = encoded_responses.data.transpose(0,1)

        if normalize:
            # normalize messages but not responses
            encoded_messages /= encoded_messages.norm(2, 0, keepdim=True)
            encoded_responses /= encoded_responses.norm(2, 0, keepdim=True)

        batch_size = encoded_messages.size(1)

        excess_memory = batch_size - (self.memory_size - self.memory_used)
        if excess_memory <= 0:
            self.encoded_messages[:, self.memory_used:self.memory_used + batch_size] = encoded_messages
            self.encoded_responses[:, self.memory_used:self.memory_used + batch_size] = encoded_responses
            self.index_map[self.memory_used:self.memory_used + batch_size] = range(self.raw_memory_used, self.raw_memory_used + batch_size)
            self.memory_used += batch_size
        else:
            print("Cap reached.")
            self.encoded_messages = torch.cat((self.encoded_messages[:, excess_memory:], encoded_messages), 1)
            self.encoded_responses = torch.cat((self.encoded_responses[:, excess_memory:], encoded_responses), 1)
            self.index_map = self.index_map[excess_memory:] + list(range(self.raw_memory_used, self.raw_memory_used + batch_size))

        self.raw_memory_used += batch_size

    def retrieve_pair(self, i):
        message = self.wd.to_words(self.raw_messages[self.index_map[i]])
        response = self.wd.to_words(self.raw_responses[self.index_map[i]])
        return message, response

    def retrieve_all(self):
        pairs = []
        for i in range(self.memory_used):
            k = self.index_map[i]
            message = self.wd.to_words(self.raw_messages[k])
            response = self.wd.to_words(self.raw_responses[k])
            pairs.append((message,response))
        return pairs

    def get_last_pair(self):
        if self.memory_used > 0:
            message = self.raw_messages[self.index_map[self.memory_used - 1]]
            response = self.raw_responses[self.index_map[self.memory_used - 1]]
            return (message, response)
        else:
            return None

    def update_encoder(self, new_encoder):
        assert len(self.raw_messages) == len(self.raw_responses)
        self.encoder = copy.deepcopy(new_encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.rnn.flatten_parameters()

        if self.memory_used > 0:
            messages = [self.raw_messages[i] for i in self.index_map[:self.memory_used]]
            responses = [self.raw_responses[i] for i in self.index_map[:self.memory_used]]

            for i in range(self.memory_used):
                self.encoded_messages[:, i] = self._encode_message(messages[i]).data
                self.encoded_responses[:, i] = self._encode_message(responses[i]).data

    def check_norms(self, verbose=False):
        msg_norms = torch.norm(self.encoded_messages[:,:self.memory_used], 2, 0)
        resp_norms = torch.norm(self.encoded_responses[:,:self.memory_used], 2, 0)

        if(verbose):
            print("Messages:")
            print("Max: ", torch.max(msg_norms))
            print("Min: ", torch.min(msg_norms))
            print("Responses:")
            print("Max: ", torch.max(resp_norms))
            print("Min: ", torch.min(resp_norms))

        return msg_norms, resp_norms

    def forward(self, input_batch, input_lengths, indices, encoded=True):
        if self.memory_used == 0:
            return None
        batch_size = input_batch.size(1)

        # Encode the input if necessary
        if not encoded:
            encoded_input = self._encode_batch(input_batch, input_lengths)
        else:
            encoded_input=input_batch.view(batch_size, self.hidden_size)

        # Convert the memory pads to Variables to track gradients
        message_pad = Variable(self.encoded_messages[:, :self.memory_used], requires_grad=False)
        response_pad = Variable(self.encoded_responses[:, :self.memory_used].transpose(0, 1), requires_grad=False)

        if USE_CUDA:
            message_pad = message_pad.cuda()
            response_pad = response_pad.cuda()

        # Compute the energies
        input_vec = encoded_input
        input_vec = self.transform(input_vec)                    # batch_size x hidden_size
        energies = torch.mm(input_vec, message_pad)                  # batch_size x memory_size

        energies *= self.temperature

        weights = F.softmax(energies, 1)

        for i in range(batch_size):
            weights[i, indices[i]] *= 0

        # Calculate the memory vector
        vector = torch.mm(weights, response_pad)                    # batch_size x hidden_size

        del message_pad, response_pad

        return vector


