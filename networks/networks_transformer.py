import torch
import torch.nn as nn

class EncoderTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, n_head=4):
        super(EncoderTransformer, self).__init__()

        self.encoder = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=n_head,  # must divides feature_size
            dim_feedforward=hidden_size,  # hidden size, 
            batch_first=True
            )

    def forward(self, input):
        """
        :param input: (seq_len, batch_size, feature_dim)
        :return:
        """
        pass


class DecoderTransformer(nn.Module):
    def __init__(self):
        super(DecoderTransformer, self).__init__()

        self.decoder = nn.TransformerEncoderLayer()
    
    def forward(self, input):
        """
        :param input:
        :return:
        """
        pass

# Transformer
##############################################################################
class TransformerAE(nn.Module):
    def __init__(self, en_input_size, de_input_size, hidden_size, n_head) -> None:
        super(TransformerAE, self).__init__()