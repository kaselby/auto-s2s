
from model import *
from preprocess import *

import torch.optim as optim

import matplotlib.pyplot as plt

class TrainingHandler(object):
    def __init__(self, model, train_set, val_set, learning_rate, save_dir, clip=5.0, tf_ratio=0.5, eps=1e-8):
        self.model=model

        self.train_set = [pair for movie in train_set for pair in movie]
        self.val_set = [pair for movie in val_set for pair in movie]

        self.clip = clip
        self.tf_ratio = tf_ratio
        self.save_dir = save_dir

        self.epoch = 0
        self.lr = learning_rate

        self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, eps=eps)

        self.train_losses = []
        self.val_losses = []

        self.mode = "auto"

    def init_memory(self, freeze=True):
        self.train_set = [convs_to_pairs(movie) for movie in self.train_lines]
        self.val_set = [convs_to_pairs(movie) for movie in self.val_lines] if self.val_lines is not None else None
        self.mode="mem"

        if freeze:
            for param in self.model.encoder.parameters():
                param.requires_grad=False

    def train_autoencoder(self, epochs, batch_size, print_interval=1, save_interval=-1):
        print("Beginning training...")
        start = time.time()

        epoch = 0
        while epoch < epochs:
            epoch += 1

            loss_total = 0.0

            batches = random_batches(batch_size, self.train_set, auto=True)
            n_batches = len(batches)

            for batch in batches:
                self.optim.zero_grad()

                # Run the train function
                loss = self.model.train_batch(batch, tf_ratio=self.tf_ratio)

                # Clip gradient norms
                c = torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

                # Update parameters with optimizers
                self.optim.step()
                loss_total += loss

            loss_avg = loss_total / n_batches

            if self.val_set is not None:
                val_loss_avg = self._val_autoencoder(batch_size)
            else:
                val_loss_avg = loss_avg

            if print_interval > 0:
                if self.epoch % print_interval == 0:
                    print_summary = '-' * 40 + '\nEPOCH #%d SUMMARY:\nTotal time spent (time left): %s, Training loss: %.4f, Validation loss: %.4f' \
                                               % (self.epoch,
                                                  time_since(start, (epoch) / epochs),
                                                  float(loss_avg), float(val_loss_avg))
                    self._print_log(print_summary)

            if self.epoch < epochs:
                if save_interval > 0:
                    if self.epoch % save_interval == 0:
                        name = "auto_" + str(self.epoch) + ".tar"
                        self._save_checkpoint(self.save_dir, name, mem=False)
            else:
                if self.save_dir is not None:
                    name = "auto_" + str(self.epoch) + ".tar"
                    self._save_checkpoint(self.save_dir, name, save_loss=True, mem=False)

    def _val_autoencoder(self, batch_size):
        total_val_loss = 0.0

        batches = random_batches(batch_size, self.val_set, auto=True)
        n_batches = len(batches)

        for batch in batches:
            loss = self.model.validate(batch)
            total_val_loss += loss

        return total_val_loss / n_batches

    def train_memory(self, epochs, batch_size, freeze_enc=False, print_interval=1, save_interval=-1):
        self.init_memory(freeze_enc)

        print("Beginning training...")
        start = time.time()

        for movie in self.train_set:
            self.model.memory.add_pairs(movie)

        epoch = 0
        while epoch < epochs:
            epoch += 1

            loss_total = 0.0

            batches = memory_random_batches(batch_size, self.train_set)
            n_batches = len(batches)

            for batch in batches:
                self.optim.zero_grad()

                # Run the train function
                loss = self.model.train_batch(batch, tf_ratio=self.tf_ratio)

                # Clip gradient norms
                c = torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

                # Update parameters with optimizers
                self.optim.step()
                loss_total += loss

            loss_avg = loss_total / n_batches

            if self.val_set is not None:
                val_loss_avg = self._val_autoencoder(batch_size)
            else:
                val_loss_avg = loss_avg

            if print_interval > 0:
                if self.epoch % print_interval == 0:
                    print_summary = '-' * 40 + '\nEPOCH #%d SUMMARY:\nTotal time spent (time left): %s, Training loss: %.4f, Validation loss: %.4f' \
                                               % (self.epoch,
                                                  time_since(start, (epoch) / epochs),
                                                  float(loss_avg), float(val_loss_avg))
                    self._print_log(print_summary)

            if self.epoch < epochs:
                if save_interval > 0:
                    if self.epoch % save_interval == 0:
                        name = "auto_" + str(self.epoch) + ".tar"
                        self._save_checkpoint(self.save_dir, name, mem=True)
            else:
                if self.save_dir is not None:
                    name = "auto_" + str(self.epoch) + ".tar"
                    self._save_checkpoint(self.save_dir, name, save_loss=True, mem=True)

    def _val_memory(self, batch_size):
        assert len(self.val_sets) > 0
        val_subsets = partition_movies(self.val_sets, self.set_size)
        total_batches = 0
        val_loss_total = 0
        for val_pairs in val_subsets:
            val_batches = memory_random_batches(batch_size, val_pairs, val_indices, val_mask)
            val_n_batches =len(val_batches)
            total_batches += val_n_batches
            for i in range(val_n_batches):
                val_loss = self.model.validate(val_batches[i])
                val_loss_total += val_loss
        val_loss_avg = val_loss_total / total_batches
        return val_loss_avg

    def _print_log(self, print_summary):
        print(print_summary)
        if self.save_dir is not None:
            save_logs(print_summary, self.save_dir)

    def _save_checkpoint(self, save_dir, name, save_loss=False, mem=False):
        # Calculate and save BLEU score
        if mem and not self.val_set is None:
            max_val_size = max([len(v) for v in self.val_set])
            self.model.memory.reset_memory(max_val_size)
            self.model.memory.update_encoder(self.model.encoder)
            old_scores, new_scores = self.model.score_set(self.val_set)
            save_scores(old_scores, new_scores, save_dir)

        if save_loss == True:
            fig_out = save_dir + FIG_FILE
            df_out = save_dir + LOSS_FILE
            if USE_CUDA:
                self._save_losses(df_out)
            else:
                (self._plot_losses()).savefig(fig_out)

        # Save model checkpoint
        self.model.memory.reset_memory()
        self.model.export_state(save_dir, name)

    def _plot_losses(self):
        fig = plt.figure()
        plt.plot(self.train_losses, color='red', label='Train_loss', marker='o')
        plt.plot(self.val_losses, color='blue', label='Val_loss', marker='o')
        plt.legend(loc='upper right', frameon=False)
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        return fig

    def _save_losses(self, path):
        outfile = open(path, 'w')
        for i in range(len(self.train_losses)):
            outfile.write(str(self.train_losses[i])+','+str(self.val_losses[i])+"\n")
        outfile.close()
