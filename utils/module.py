"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           module.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import collections

class ModelManager(nn.Module):

    def __init__(self, args, num_word, num_slot, num_intent, lattice_embed, bigram_embed, hidden_size, label_size,
                 num_heads, num_layers,
                 use_abs_pos, use_rel_pos, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 ff_size=-1, scaled=True, dropout=None, use_bigram=True, mode=collections.defaultdict(bool),
                 dvc=None, vocabs=None,
                 rel_pos_shared=True, max_seq_len=-1, k_proj=True, q_proj=True, v_proj=True, r_proj=True,
                 self_supervised=False, attn_ff=True, pos_norm=False, ff_activate='relu', rel_pos_init=0,
                 abs_pos_fusion_func='concat', embed_dropout_pos='0',
                 four_pos_shared=True, four_pos_fusion=None, four_pos_fusion_shared=True,
                 bert_embedding=None, use_pytorch_dropout=False):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # Initialize an embedding object.
        self.__embedding = EmbeddingCollection(
            self.__num_word,
            self.__args.word_embedding_dim
        )

        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(
            self.__args.word_embedding_dim,
            self.__args.encoder_hidden_dim,
            self.__args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim,
            self.__args.attention_hidden_dim,
            self.__args.attention_output_dim,
            self.__args.dropout_rate
        )

        # Initialize an Decoder object for intent.
        self.__intent_decoder = LSTMDecoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.intent_decoder_hidden_dim,
            self.__num_intent, self.__args.dropout_rate,
            embedding_dim=self.__args.intent_embedding_dim
        )
        # Initialize an Decoder object for slot.
        # todo: this should be modified
        self.__slot_decoder = LSTMDecoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot, self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim,
            extra_dim=self.__num_intent
        )

        # One-hot encoding for augment data feed. 
        self.__intent_embedding = nn.Embedding(
            self.__num_intent, self.__num_intent
        )
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)
        self.__intent_embedding.weight.requires_grad = False

        self.use_pytorch_dropout = use_pytorch_dropout
        self.four_pos_fusion_shared = four_pos_fusion_shared
        self.mode = mode
        self.four_pos_shared = four_pos_shared
        self.abs_pos_fusion_func = abs_pos_fusion_func
        self.lattice_embed = lattice_embed
        self.bigram_embed = bigram_embed
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        # self.relative_position = relative_position
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.rel_pos_shared = rel_pos_shared
        self.self_supervised = self_supervised
        self.vocabs = vocabs
        self.attn_ff = attn_ff
        self.pos_norm = pos_norm
        self.ff_activate = ff_activate
        self.rel_pos_init = rel_pos_init
        self.embed_dropout_pos = embed_dropout_pos

        # if self.relative_position:
        #     print('现在还不支持相对编码！')
        #     exit(1208)

        # if self.add_position:
        #     print('暂时只支持位置编码的concat模式')
        #     exit(1208)

        if self.use_rel_pos and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.pe = None

        if self.use_abs_pos:
            self.abs_pos_encode = Absolute_SE_Position_Embedding(self.abs_pos_fusion_func,
                                                                 self.hidden_size, learnable=self.learnable_position,
                                                                 mode=self.mode,
                                                                 pos_norm=self.pos_norm)

        if self.use_rel_pos:
            pe = get_embedding(max_seq_len, hidden_size, rel_pos_init=self.rel_pos_init)
            pe_sum = pe.sum(dim=-1, keepdim=True)
            if self.pos_norm:
                with torch.no_grad():
                    pe = pe / pe_sum
            self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
            if self.four_pos_shared:
                self.pe_ss = self.pe
                self.pe_se = self.pe
                self.pe_es = self.pe
                self.pe_ee = self.pe
            else:
                self.pe_ss = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
                self.pe_se = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
                self.pe_es = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
                self.pe_ee = nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        else:
            self.pe = None
            self.pe_ss = None
            self.pe_se = None
            self.pe_es = None
            self.pe_ee = None

        # if self.add_position:
        #     print('现在还不支持位置编码通过concat的方式加入')
        #     exit(1208)

        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        if ff_size == -1:
            ff_size = self.hidden_size
        self.ff_size = ff_size
        self.scaled = scaled
        if dvc == None:
            dvc = 'cpu'
        self.dvc = torch.device(dvc)
        if dropout is None:
            self.dropout = collections.defaultdict(int)
        else:
            self.dropout = dropout
        self.use_bigram = use_bigram

        if self.use_bigram:
            self.bigram_size = self.bigram_embed.embedding.weight.size(1)
            self.char_input_size = self.lattice_embed.embedding.weight.size(
                1) + self.bigram_embed.embedding.weight.size(1)
        else:
            self.char_input_size = self.lattice_embed.embedding.weight.size(1)

        self.lex_input_size = self.lattice_embed.embedding.weight.size(1)

        if use_pytorch_dropout:
            self.embed_dropout = nn.Dropout(self.dropout['embed'])
            self.gaz_dropout = nn.Dropout(self.dropout['gaz'])
            self.output_dropout = nn.Dropout(self.dropout['output'])
        else:
            self.embed_dropout = MyDropout(self.dropout['embed'])
            self.gaz_dropout = MyDropout(self.dropout['gaz'])
            self.output_dropout = MyDropout(self.dropout['output'])

        self.char_proj = nn.Linear(self.char_input_size, self.hidden_size)
        self.lex_proj = nn.Linear(self.lex_input_size, self.hidden_size)

        self.encoder = Transformer_Encoder(self.hidden_size, self.num_heads, self.num_layers,
                                           relative_position=self.use_rel_pos,
                                           learnable_position=self.learnable_position,
                                           add_position=self.add_position,
                                           layer_preprocess_sequence=self.layer_preprocess_sequence,
                                           layer_postprocess_sequence=self.layer_postprocess_sequence,
                                           dropout=self.dropout,
                                           scaled=self.scaled,
                                           ff_size=self.ff_size,
                                           mode=self.mode,
                                           dvc=self.dvc,
                                           max_seq_len=self.max_seq_len,
                                           pe=self.pe,
                                           pe_ss=self.pe_ss,
                                           pe_se=self.pe_se,
                                           pe_es=self.pe_es,
                                           pe_ee=self.pe_ee,
                                           k_proj=self.k_proj,
                                           q_proj=self.q_proj,
                                           v_proj=self.v_proj,
                                           r_proj=self.r_proj,
                                           attn_ff=self.attn_ff,
                                           ff_activate=self.ff_activate,
                                           lattice=True,
                                           four_pos_fusion=self.four_pos_fusion,
                                           four_pos_fusion_shared=self.four_pos_fusion_shared,
                                           use_pytorch_dropout=self.use_pytorch_dropout)

        self.output = nn.Linear(self.hidden_size, self.label_size)
        if self.self_supervised:
            self.output_self_supervised = nn.Linear(self.hidden_size, len(vocabs['char']))
            print('self.output_self_supervised:{}'.format(self.output_self_supervised.weight.size()))
        self.crf = get_crf_zero_init(self.label_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
        self.batch_num = 0

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print('Model parameters are listed as follows:\n')

        print('\tnumber of word:                            {};'.format(self.__num_word))
        print('\tnumber of slot:                            {};'.format(self.__num_slot))
        print('\tnumber of intent:						    {};'.format(self.__num_intent))
        print('\tword embedding dimension:				    {};'.format(self.__args.word_embedding_dim))
        print('\tencoder hidden dimension:				    {};'.format(self.__args.encoder_hidden_dim))
        print('\tdimension of intent embedding:		    	{};'.format(self.__args.intent_embedding_dim))
        print('\tdimension of slot embedding:			    {};'.format(self.__args.slot_embedding_dim))
        print('\tdimension of slot decoder hidden:  	    {};'.format(self.__args.slot_decoder_hidden_dim))
        print('\tdimension of intent decoder hidden:        {};'.format(self.__args.intent_decoder_hidden_dim))
        print('\thidden dimension of self-attention:        {};'.format(self.__args.attention_hidden_dim))
        print('\toutput dimension of self-attention:        {};'.format(self.__args.attention_output_dim))

        print('\nEnd of parameters show. Now training begins.\n\n')

    def forward(self, text, seq_lens, lattice, bigrams, seq_len, lex_num, pos_s, pos_e,
                target, chars_target=None, n_predicts=None, forced_slot=None, forced_intent=None):
        word_tensor, _ = self.__embedding(text)

        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        # transformer_hiddens = self.__transformer(pos_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        pred_intent = self.__intent_decoder(
            hiddens, seq_lens,
            forced_input=forced_intent
        )

        if not self.__args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)
            feed_intent = self.__intent_embedding(idx_intent.squeeze(1))
        else:
            feed_intent = pred_intent

        pred_slot = self.__slot_decoder(
            hiddens, seq_lens,
            forced_input=forced_slot,
            extra_input=feed_intent
        )

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()

    def lattice_forward(self, hiddens, seq_lens, lattice, bigrams, seq_len, lex_num, pos_s, pos_e,
                target, chars_target=None):
        if self.mode['debug']:
            print('lattice:{}'.format(lattice))
            print('bigrams:{}'.format(bigrams))
            print('seq_len:{}'.format(seq_len))
            print('lex_num:{}'.format(lex_num))
            print('pos_s:{}'.format(pos_s))
            print('pos_e:{}'.format(pos_e))

        batch_size = lattice.size(0)
        max_seq_len_and_lex_num = lattice.size(1)
        max_seq_len = bigrams.size(1)

        raw_embed = self.lattice_embed(lattice)
        # raw_embed 是字和词的pretrain的embedding，但是是分别trian的，所以需要区分对待
        if self.use_bigram:
            bigrams_embed = self.bigram_embed(bigrams)
            bigrams_embed = torch.cat([bigrams_embed,
                                       torch.zeros(size=[batch_size, max_seq_len_and_lex_num - max_seq_len,
                                                         self.bigram_size]).to(bigrams_embed)], dim=1)
            raw_embed_char = torch.cat([raw_embed, bigrams_embed], dim=-1)
        else:
            raw_embed_char = raw_embed

        dim2 = 0
        dim3 = 2
        # print('raw_embed:{}'.format(raw_embed[:,dim2,:dim3]))
        # print('raw_embed_char:{}'.format(raw_embed_char[:, dim2, :dim3]))
        if self.embed_dropout_pos == '0':
            raw_embed_char = self.embed_dropout(raw_embed_char)
            raw_embed = self.gaz_dropout(raw_embed)
        # print('raw_embed_dropout:{}'.format(raw_embed[:,dim2,:dim3]))
        # print('raw_embed_char_dropout:{}'.format(raw_embed_char[:, dim2, :dim3]))

        embed_char = self.char_proj(raw_embed_char) # Linear
        if self.mode['debug']:
            print('embed_char:{}'.format(embed_char[:2]))
        char_mask = seq_len_to_mask(seq_len, max_len=max_seq_len_and_lex_num).bool()
        # if self.embed_dropout_pos == '1':
        #     embed_char = self.embed_dropout(embed_char)
        embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

        embed_lex = self.lex_proj(raw_embed)
        if self.mode['debug']:
            print('embed_lex:{}'.format(embed_lex[:2]))
        # if self.embed_dropout_pos == '1':
        #     embed_lex = self.embed_dropout(embed_lex)

        lex_mask = (seq_len_to_mask(seq_len + lex_num).bool() ^ char_mask.bool())
        embed_lex.masked_fill_(~(lex_mask).unsqueeze(-1), 0)

        assert char_mask.size(1) == lex_mask.size(1)

        embedding = embed_char + embed_lex
        if self.mode['debug']:
            print('embedding:{}'.format(embedding[:2]))

        if self.embed_dropout_pos == '1':
            embedding = self.embed_dropout(embedding)

        if self.use_abs_pos:
            embedding = self.abs_pos_encode(embedding, pos_s, pos_e)

        if self.embed_dropout_pos == '2':
            embedding = self.embed_dropout(embedding)
        # embedding = self.embed_dropout(embedding)

        # print('embedding:{}'.format(embedding[:,dim2,:dim3]))

        if self.batch_num == 327:
            print('{} embed:{}'.format(self.batch_num,
                                       embedding[:2, dim2, :dim3]))

        encoded = self.encoder(embedding, seq_len, lex_num=lex_num, pos_s=pos_s, pos_e=pos_e,
                               print_=(self.batch_num == 327))

        if self.batch_num == 327:
            print('{} encoded:{}'.format(self.batch_num,
                                         encoded[:2, dim2, :dim3]))

        if hasattr(self, 'output_dropout'):
            encoded = self.output_dropout(encoded)

        encoded = encoded[:, :max_seq_len, :]
        pred = self.output(encoded)

        if self.batch_num == 327:
            print('{} pred:{}'.format(self.batch_num,
                                      pred[:2, dim2, :dim3]))

        # print('pred:{}'.format(pred[:,dim2,:dim3]))
        # exit()

        mask = seq_len_to_mask(seq_len).bool()

        if self.mode['debug']:
            print('debug mode:finish!')
            exit(1208)
        if self.training:
            loss = self.crf(pred, target, mask).mean(dim=0)
            if self.self_supervised:
                # print('self supervised loss added!')
                chars_pred = self.output_self_supervised(encoded)
                chars_pred = chars_pred.view(size=[batch_size * max_seq_len, -1])
                chars_target = chars_target.view(size=[batch_size * max_seq_len])
                self_supervised_loss = self.loss_func(chars_pred, chars_target)
                # print('self_supervised_loss:{}'.format(self_supervised_loss))
                # print('supervised_loss:{}'.format(loss))
                loss += self_supervised_loss

            if self.batch_num == 327:
                print('{} loss:{}'.format(self.batch_num, loss))
                exit()

            # exit()
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}
            if self.self_supervised:
                chars_pred = self.output_self_supervised(encoded)
                result['chars_pred'] = chars_pred

            return result




    def golden_intent_predict_slot(self, text, seq_lens, golden_intent, n_predicts=1):
        word_tensor, _ = self.__embedding(text)
        embed_intent = self.__intent_embedding(golden_intent)

        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        pred_slot = self.__slot_decoder(
            hiddens, seq_lens, extra_input=embed_intent
        )
        _, slot_index = pred_slot.topk(n_predicts, dim=-1)

        # Just predict single slot value.
        return slot_index.cpu().data.numpy().tolist()


class EmbeddingCollection(nn.Module):
    """
    Provide word vector and position vector encoding.
    """

    def __init__(self, input_dim, embedding_dim, max_len=5000):
        super(EmbeddingCollection, self).__init__()

        self.__input_dim = input_dim
        # Here embedding_dim must be an even embedding.
        self.__embedding_dim = embedding_dim
        self.__max_len = max_len

        # Word vector encoder.
        self.__embedding_layer = nn.Embedding(
            self.__input_dim, self.__embedding_dim
        )

        # Position vector encoder.
        # self.__position_layer = torch.zeros(self.__max_len, self.__embedding_dim)
        # position = torch.arange(0, self.__max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.__embedding_dim, 2) *
        #                      (-math.log(10000.0) / self.__embedding_dim))

        # Sine wave curve design.
        # self.__position_layer[:, 0::2] = torch.sin(position * div_term)
        # self.__position_layer[:, 1::2] = torch.cos(position * div_term)
        #
        # self.__position_layer = self.__position_layer.unsqueeze(0)
        # self.register_buffer('pe', self.__position_layer)

    def forward(self, input_x):
        # Get word vector encoding.
        embedding_x = self.__embedding_layer(input_x)

        # Get position encoding.
        # position_x = Variable(self.pe[:, :input_x.size(1)], requires_grad=False)

        # Board-casting principle.
        return embedding_x, embedding_x


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return torch.cat([padded_hiddens[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1
        )
        self.__linear_layer = nn.Linear(
            self.__hidden_dim,
            self.__output_dim
        )

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """

        # Concatenate information tensor if possible.
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is None or forced_input is not None:

            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                # Segment input hidden tensors.
                seg_hiddens = input_tensor[sent_start_pos: sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos: sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(seq_lens[sent_i], -1)
                        seg_prev_tensor = torch.cat([self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0)
                    else:
                        seg_prev_tensor = self.__init_tensor

                    # Concatenate forced target tensor.
                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)

                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
                linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i], -1))

                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                # It's necessary to remember h and c state
                # when output prediction every single step.
                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                    lstm_out = self.__linear_layer(lstm_out.view(1, -1))
                    output_tensor_list.append(lstm_out)

                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.__hidden_dim), dim=-1)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim, self.__input_dim, self.__input_dim,
            self.__hidden_dim, self.__output_dim, self.__dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(
            dropout_x, dropout_x, dropout_x
        )

        flat_x = torch.cat(
            [attention_x[i][:seq_lens[i], :] for
             i in range(0, len(seq_lens))], dim=0
        )
        return flat_x
