import math
import random
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EncoderTransformer(nn.Module):
    def __init__(self, input_size, n_head, hidden_size, n_layer):
        super(EncoderTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=n_head,  # must divides feature_size
            dim_feedforward=hidden_size,  # hidden size,
            )
        encoder_norm = nn.LayerNorm(input_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layer, encoder_norm)

    def forward(self, input, src_mask=None, src_key_padding_mask=None):
        """
        :param input: (seq_len, batch_size, feature_dim)
        :return:
            output: (seq_len, batch_size, feature_dim)
        """
        input = self.pos_encoder(input)
        output = self.transformer_encoder(input, src_mask, src_key_padding_mask)
        return output


class DecoderTransformer(nn.Module):
    def __init__(self, input_size, n_head, hidden_size, n_layer):
        super(DecoderTransformer, self).__init__()
        self.input_size = input_size
        self.n_units_hidden1 = 256
        self.n_units_hidden2 = 256

        self.init_input = self.initInput()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_size,
            nhead=n_head,
            dim_feedforward=hidden_size,
        )
        decoder_norm = nn.LayerNorm(input_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layer, decoder_norm)
        self.linear1 = nn.Sequential(nn.Linear(input_size, self.n_units_hidden1),
                                     nn.LeakyReLU(True),
                                     nn.Linear(self.n_units_hidden1, input_size - 6),
                                     # nn.Sigmoid()
                                     )
        self.linear2 = nn.Sequential(nn.Linear(input_size, self.n_units_hidden2),
                                     nn.ReLU(True),
                                     nn.Dropout(0.2),
                                     nn.Linear(self.n_units_hidden2, 6),
                                     # nn.Sigmoid()
                                     )
        self.linear3 = nn.Sequential(nn.Linear(input_size, self.n_units_hidden2),
                                     nn.ReLU(True),
                                     nn.Dropout(0.2),
                                     nn.Linear(self.n_units_hidden2, 1),
                                     nn.Sigmoid()
                                     )

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        :param tgt: (n_parts, batch_size, feat_dim)
        :param memory: (n_parts, batch_size, feat_dim)
        :param tgt_mask: (n_parts, n_parts)
        :param tgt_key_padding_mask: (batch_size, n_parts)
        :return:
            output_seq: (n_parts, batch, output_size)
            stop_sign: (n_parts, batch, 1)
        """
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output_code = self.linear1(output)
        output_param = self.linear2(output)
        stop_sign = self.linear3(output)
        output_seq = torch.cat([output_code, output_param], dim=2)

        return output_seq, stop_sign

    def initInput(self):
        initial_code = torch.zeros((1, 1, self.input_size - 6), requires_grad=False)
        initial_param = torch.tensor([0.5, 0.5, 0.5, 1, 1, 1], dtype=torch.float32, requires_grad=False).unsqueeze(0).unsqueeze(0)
        initial = torch.cat([initial_code, initial_param], dim=2)
        return initial


# Transformer
##############################################################################
class TransformerAE(nn.Module):
    def __init__(self, en_input_size, de_input_size, hidden_size, n_head=2):
        super(TransformerAE, self).__init__()
        self.n_layer = 3
        self.encoder = EncoderTransformer(en_input_size, n_head, hidden_size, self.n_layer)
        self.decoder = DecoderTransformer(de_input_size, n_head, hidden_size, self.n_layer)
        self.max_length = 10

    def infer_encoder(self, input_seq, batch_n_parts=None):
        """
        :param input_seq: (n_parts, batch_size, feature_dim)
        :return:
            memory: (n_parts, batch, feature_dim)
        """
        if batch_n_parts is None:
            src_key_padding_mask = None
        else:
            input_len = input_seq.size(0)
            src_key_padding_mask = self.get_key_padding_mask(input_len, batch_n_parts).cuda()

        input_seq = input_seq[:,:,:-6] # important: remove cond to have the same en_feat_dim and de_feat_dim
        memory = self.encoder(input_seq, src_key_padding_mask=src_key_padding_mask)
        return memory

    def infer_decoder(self, target_seq, memory, batch_n_parts):
        # For training Transformer, we always use target_seq as the next step input
        # https://datascience.stackexchange.com/questions/104179/is-the-transformer-decoder-an-autoregressive-model?noredirect=1&lq=1
        batch_size = target_seq.size(1)
        target_len = target_seq.size(0)

        # the input to the decoder is the target seq shifted one position to the right by the start token
        # https://datascience.stackexchange.com/questions/88981/what-are-the-inputs-to-the-first-decoder-layer-in-a-transformer-model-during-the
        decoder_input = self.decoder.init_input.detach().repeat(1, batch_size, 1).cuda()
        target_seq = torch.cat([decoder_input, target_seq], dim=0)
        target_seq = target_seq[:-1, :, :]

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len).cuda() # (target_len, target_len)
        tgt_key_padding_mask = self.get_key_padding_mask(target_len, batch_n_parts)  # (batch_size, target_len)
        tgt_key_padding_mask = torch.cat([torch.zeros((batch_size, 1)), tgt_key_padding_mask], dim=1).cuda()
        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

        output_seq, stop_sign = self.decoder(target_seq, memory, tgt_mask=tgt_mask,
                                             tgt_key_padding_mask=tgt_key_padding_mask)
        return output_seq, stop_sign
    
    def infer_decoder_stop(self, memory, length=None):
        # At inference time, we use decoder's predictions as the next step input
        decoder_outputs = []
        stop_signs = []
        target_seq = self.decoder.init_input.detach().repeat(1, 1, 1).cuda()

        for di in range(self.max_length):
            output_seq, stop_sign = self.decoder(target_seq, memory)
            decoder_outputs.append(output_seq[-1, :, :])
            stop_signs.append(stop_sign[-1, :, :])

            if length is not None:
                if di == length - 1:
                    break
            elif stop_sign[-1, 0, 0] > 0.5:
                # stop condition
                break
            target_seq = torch.cat([target_seq, output_seq[-1, :, :].unsqueeze(0)])
        decoder_outputs = torch.stack(decoder_outputs, dim=0)
        stop_signs = torch.stack(stop_signs, dim=0)
        return target_seq, stop_sign


    def forward(self, input_seq, target_seq, batch_n_parts):
        """
        :param input_seq: (seq_len, batch_size, feature_dim)
        :param target_seq: (seq_len, batch_size, feature_dim)
        :param batch_n_parts: list of `batch_size` int
        :return:
            decoder_outputs: (seq_len, batch, )
            stop_signs: (seq_len, batch, )
        """

        memory = self.infer_encoder(input_seq, batch_n_parts)
        output_seq, output_stop = self.infer_decoder(target_seq, memory, batch_n_parts)
        return output_seq, output_stop

    @staticmethod
    def get_key_padding_mask(seq_len, batch_n_parts):
        mask = torch.arange(seq_len).unsqueeze(0) >= torch.tensor(batch_n_parts).unsqueeze(1)
        mask = mask.float()
        mask[mask == 1] = float('-inf')
        return mask


if __name__ == '__main__':
    pass