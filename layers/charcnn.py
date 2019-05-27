from torch import nn
import torch.nn.functional as F
import torch
import logging


class CharCNN(nn.Module):

    def __init__(self, dropout_rate, char_embed_dim, hidden_dim):
        super(CharCNN, self).__init__()
        logger = logging.getLogger("ner")
        logger.info("Build Char CNN layer...")

        self.dropout_rate = dropout_rate
        self.char_embed_dim = char_embed_dim
        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(self.dropout_rate)

        self.char_cnn_filters = []
        self.filter_kernels = [3]
        for idx, k_size in enumerate(self.filter_kernels):
            char_cnn = nn.Conv1d(self.char_embed_dim, self.hidden_dim, kernel_size=k_size, padding=1)
            self.char_cnn_filters.append(char_cnn)
            setattr(self, 'char_cnn_' + str(idx), char_cnn)

    def forward(self, char_embeddings, max_pool=False):
        batch_size = char_embeddings.size(0)
        max_len = char_embeddings.size(1)
        max_char_len = char_embeddings.size(2)
        embedding_dim = char_embeddings.size(-1)

        char_embeddings = self.dropout(char_embeddings).view(-1, max_char_len, embedding_dim)
        feature_outs = []
        for idx in range(len(self.filter_kernels)):
            feature_map = self.char_cnn_filters[idx](char_embeddings.transpose(2, 1))
            if max_pool:
                char_cnn_out = F.max_pool1d(feature_map, kernel_size=max_char_len).transpose(2, 1)
            else:
                char_cnn_out = feature_map
            feature_outs.append(char_cnn_out.view(batch_size, max_len, -1))
        char_cnn_out = torch.cat(feature_outs, -1)
        return char_cnn_out
