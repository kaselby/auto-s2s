from utils import *

class Beam(object):
    def __init__(self, tokens, score, state):
        self.tokens=tokens
        self.score=score
        self.state=state

    def update(self, token, score, state):
        return Beam(self.tokens+[token], self.score*score, state)

    def stop(self, max_length=10):
        return len(self.tokens) >= max_length or EOS_INDEX in self.tokens


class BeamIterator(object):
    def __init__(self, model, initial_state, beam_width, num_beams, max_length=10):
        self.model = model
        self.initial_state = initial_state
        self.beam_width = beam_width
        self.num_beams = num_beams
        self.max_length = max_length

    def print_beams(self,beams):
        for beam in beams:
            print(self.model.wd.to_words(beam.tokens))

    def stop_search(self, beams):
        for beam in beams:
            if not beam.stop():
                return False
        return True

    def search(self):
        stop=False
        beams = [Beam([SOS_INDEX], 1., self.initial_state)]
        while not stop:
            beams = self._search_iter(beams)
            if self.stop_search(beams):
                stop = True
        return beams[0].tokens

    def _search_iter(self, beams):
        current_beams = []
        for beam in beams:
            if not beam.stop():
                beam_input = beam.tokens[-1]
                decoder_input = Variable(torch.LongTensor([beam_input]))
                if USE_CUDA:
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = self.model.decoder(decoder_input, beam.state, lengths=None)
                output_probs = F.softmax(decoder_output, dim=1)

                topv, topi = output_probs.data.topk(self.beam_width)
                for j in range(self.beam_width):
                    current_beams.append(beam.update(topi[0,j], topv[0,j], decoder_hidden))
            else:
                current_beams.append(beam)

        sorted_beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        top_beams = sorted_beams[:self.num_beams]

        return top_beams

