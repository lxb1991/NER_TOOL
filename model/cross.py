import torch
from torch.nn import Module, Embedding, Linear, Dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM
from layers.charcnn import CharCNN
from layers.no_param_lstm import NoParamLSTM
import numpy as np


class CrossNER(Module):

    def __init__(self, hyper_param, word_embedding, char_embedding, vocabs,
                 task_vocab_size, domain_vocab_size, param_types, device):
        super().__init__()

        word_embeddings_weight = torch.FloatTensor(word_embedding)
        self.word_matrix = Embedding.from_pretrained(word_embeddings_weight, freeze=False)

        char_embeddings_weight = torch.FloatTensor(char_embedding)
        self.char_matrix = Embedding.from_pretrained(char_embeddings_weight, freeze=False)
        self.char_cnn = CharCNN(hyper_param.drop_out, hyper_param.char_embed_dim, hyper_param.char_cnn_kernels)

        self.task_embeddings = Embedding.from_pretrained(
            torch.from_numpy(self.random_embedding(task_vocab_size, 8)), freeze=False).float()
        self.domain_embeddings = Embedding.from_pretrained(
            torch.from_numpy(self.random_embedding(domain_vocab_size, 8)), freeze=False).float()

        self.lstm_input_size = hyper_param.word_embed_dim + hyper_param.char_cnn_kernels * 3
        self.rnn = NoParamLSTM(bidirectional=True, num_layers=1, input_size=self.lstm_input_size,
                               hidden_size=hyper_param.lstm_hidden, batch_first=True)

        self.drop_out = Dropout(p=hyper_param.drop_out)
        self.fc = {}
        for key in param_types:
            self.fc[key] = Linear(hyper_param.lstm_hidden * 2, len(vocabs[key]))
            setattr(self, 'fc_' + '_'.join(key), self.fc[key])
        self.device = device

    @staticmethod
    def random_embedding(vocab_size, dimension):
        pretrain_emb = np.zeros([vocab_size, dimension])
        scale = np.sqrt(3.0 / dimension)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, dimension])
        return pretrain_emb

    def forward(self, word_idx, char_idx, word_lens, tags_idx, forward_idx, backward_idx, tokens_mask,
                task_idx, domain_idx, param_type):

        word_idx = word_idx.to(self.device)
        char_idx = char_idx.to(self.device)
        if tags_idx is not None:
            tags_idx = tags_idx.to(self.device)
        if forward_idx is not None and backward_idx is not None:
            forward_idx = forward_idx.to(self.device)
            backward_idx = backward_idx.to(self.device)
        word_lens = word_lens.to(self.device)
        tokens_mask = tokens_mask.to(self.device)
        task_idx = task_idx.to(self.device)
        domain_idx = domain_idx.to(self.device)

        word_emb = self.word_matrix(word_idx)
        char_emb = self.char_matrix(char_idx)

        wc_emb = torch.cat((word_emb, self.char_cnn(char_emb)), -1)
        word_encode = self.drop_out(wc_emb)

        # pack_word = pack_padded_sequence(word_encode, word_lens, batch_first=True)
        hidden_state = None
        rnn_encode, hidden_state = self.rnn(word_encode, hidden_state,
                                            self.task_embeddings(task_idx), self.domain_embeddings(domain_idx))
        # pad_encode, _ = pad_packed_sequence(rnn_encode, batch_first=True)

        if "ner" == param_type[0]:
            word_score = self.fc[param_type](rnn_encode)
            label_hat = word_score.argmax(-1)
            return word_score, tags_idx, label_hat, tokens_mask
        else:
            final_forward, final_backward = rnn_encode.chunk(2, -1)
            forward_idx = torch.masked_select(forward_idx, tokens_mask.byte())
            final_forward = torch.masked_select(final_forward, tokens_mask.unsqueeze(-1).byte())
            backward_idx = torch.masked_select(backward_idx, tokens_mask.byte())
            final_backward = torch.masked_select(final_backward, tokens_mask.unsqueeze(-1).byte())
            return final_forward, final_backward, forward_idx, backward_idx, word_lens

