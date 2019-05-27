import torch
from torch.nn import Module, LSTM, Embedding, Linear, Dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from layers.charcnn import CharCNN


class BaseNER(Module):

    def __init__(self, hyper_param, word_embedding, char_embedding, label_vocab_size, device):
        super(BaseNER, self).__init__()
        self.device = device

        word_embeddings_weight = torch.FloatTensor(word_embedding)
        self.word_matrix = Embedding.from_pretrained(word_embeddings_weight, freeze=False)

        char_embeddings_weight = torch.FloatTensor(char_embedding)
        self.char_matrix = Embedding.from_pretrained(char_embeddings_weight, freeze=False)
        self.char_cnn = CharCNN(hyper_param.drop_out, hyper_param.char_embed_dim, hyper_param.char_cnn_kernels)

        self.lstm_input_size = hyper_param.word_embed_dim + hyper_param.char_cnn_kernels * 3
        self.lstm_input_size = hyper_param.word_embed_dim + 240
        self.lstm = LSTM(self.lstm_input_size, hyper_param.lstm_hidden,
                         batch_first=True, bidirectional=True)

        self.drop_out = Dropout(p=hyper_param.drop_out)
        self.project_input_size = hyper_param.lstm_hidden * 2
        self.project = Linear(self.project_input_size, label_vocab_size)

    def forward(self, word_idx, char_idx, word_lens, tag_idx):
        word_idx = word_idx.to(self.device)
        char_idx = char_idx.to(self.device)
        tag_idx = tag_idx.to(self.device)

        word_emb = self.word_matrix(word_idx)
        char_emb = self.char_matrix(char_idx)

        wc_emb = torch.cat((word_emb, self.char_cnn(char_emb)), -1)
        word_encode = self.drop_out(wc_emb)

        pack_word = pack_padded_sequence(word_encode, word_lens, batch_first=True)
        lstm_encode, _ = self.lstm(pack_word)
        pad_encode, _ = pad_packed_sequence(lstm_encode, batch_first=True)

        final_encode = self.drop_out(pad_encode)
        final_score = self.project(final_encode)
        final_hat = torch.argmax(final_score, -1)
        return final_score, final_hat, tag_idx


class BaseLM(Module):

    def __init__(self, hyper_param, word_embedding, char_embedding, label_vocab_size, device):
        super(BaseLM, self).__init__()
        self.device = device

        word_embeddings_weight = torch.FloatTensor(word_embedding)
        self.word_matrix = Embedding.from_pretrained(word_embeddings_weight, freeze=False)

        char_embeddings_weight = torch.FloatTensor(char_embedding)
        self.char_matrix = Embedding.from_pretrained(char_embeddings_weight, freeze=False)
        self.char_cnn = CharCNN(hyper_param.drop_out, hyper_param.char_embed_dim, hyper_param.char_cnn_kernels)

        self.lstm_input_size = hyper_param.word_embed_dim + hyper_param.char_cnn_kernels * 3
        self.lstm = LSTM(self.lstm_input_size, hyper_param.lstm_hidden,
                         batch_first=True, bidirectional=True)
        self.drop_out = Dropout(p=hyper_param.drop_out)

    def forward(self, word_idx, char_idx, forward_idx, backward_idx, tokens_mask, word_len):
        word_idx = word_idx.to(self.device)
        char_idx = char_idx.to(self.device)
        forward_idx = forward_idx.to(self.device)
        backward_idx = backward_idx.to(self.device)
        word_len = word_len.to(self.device)
        tokens_mask = tokens_mask.to(self.device)

        word_emb = self.word_matrix(word_idx)
        char_emb = self.char_matrix(char_idx)

        wc_emb = torch.cat((word_emb, self.char_cnn(char_emb)), -1)
        word_encode = self.drop_out(wc_emb)

        pack_word = pack_padded_sequence(word_encode, word_len, batch_first=True)
        lstm_encode, _ = self.lstm(pack_word)
        pad_encode, _ = pad_packed_sequence(lstm_encode, batch_first=True)

        final_forward, final_backward = pad_encode.chunk(2, -1)

        forward_idx = torch.masked_select(forward_idx, tokens_mask.byte())
        final_forward = torch.masked_select(final_forward, tokens_mask.unsqueeze(-1).byte())
        backward_idx = torch.masked_select(backward_idx, tokens_mask.byte())
        final_backward = torch.masked_select(final_backward, tokens_mask.unsqueeze(-1).byte())

        return final_forward, final_backward, forward_idx, backward_idx, word_len
