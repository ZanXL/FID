
import torch
import copy
from torch import nn
import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as fn


class FID(SequentialRecommender):

    def __init__(self, config, dataset):
        super(FID, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.k_interests = config['k_interests']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.max_seq_length = config['max_seq_length']
        self.time_span = config['time_span']
        self.timestamp = config['TIME_FIELD'] + '_list'
        self.time_bins = config['time_bins']
        self.c=config['c']
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.seq_len = self.max_seq_length
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.tset=self.time_bins.split(",");

        self.time_matrix_emb = nn.Embedding(self.time_span + 1, self.hidden_size, padding_idx=0)

        self.trm_encoder = MFTransformerEncoder(n_layers=self.n_layers, n_heads=self.n_heads,
                                                   k_interests=self.k_interests, hidden_size=self.hidden_size,
                                                   seq_len=self.seq_len,
                                                   inner_size=self.inner_size,
                                                   hidden_dropout_prob=self.hidden_dropout_prob,
                                                   attn_dropout_prob=self.attn_dropout_prob,
                                                   hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.mlp_layers = 3
        self.disen_interest = MaskedItemToInterestAggregation(self.hidden_size,
                                                              self.mlp_layers,
                                                              self.k_interests)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def embedding_layer(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)     #lg:a = torch.tensor([[1,2,3], [4,5,6]]);  a.size(1))   # 第1维有1，2，3（或4，5，6）三个数据   #position_ids:tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        return item_emb, position_embedding

    def forward(self, item_seq, item_seq_len, time_f_w_matrix):

        item_emb, position_embedding = self.embedding_layer(item_seq)
        item_emb = self.LayerNorm(item_emb)
        item_emb = self.dropout(item_emb)
        item_mask = (item_seq != 0).unsqueeze(-1).repeat(1, 1, item_emb.shape[-1])

        trm_output = self.trm_encoder(time_f_w_matrix,
                                      item_emb,
                                      item_mask,
                                      position_embedding,
                                      output_all_encoded_layers=True)
        #output = trm_output[-1]
        # U = self.disen_interest(output, item_mask, time_f_w_matrix)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        return output # [B H]

    def get_time_matrix(self, time_seq):  # time_seq -> time_matrix: [B, L] -> [B, L, L]

        time_matrix_i = time_seq.unsqueeze(-1).expand([-1, self.max_seq_length, self.max_seq_length])
        time_matrix_j = time_seq.unsqueeze(1).expand([-1, self.max_seq_length, self.max_seq_length])
        time_matrix = torch.abs(time_matrix_i - time_matrix_j)
        time_matrix=time_matrix/(24*60*60)                              #Unit / days

        max_time_matrix = (torch.ones_like(time_matrix) * self.time_span).to(self.device)
        time_matrix = torch.where(time_matrix > self.time_span, max_time_matrix, time_matrix)
        time_matrix = time_matrix.float()
        time_f_w_matrix = torch.zeros_like(time_matrix, dtype=torch.float)
        ones = torch.ones_like(time_matrix, dtype=torch.float)

        conf = []
        for i in self.tset:
            conf.append(int(i))
        all_mask = []
        all_mask_weight = []
        for i in range(len(conf)):
            if i == len(conf) - 1:
                t = conf[i]
                f_t = math.exp(-self.c * t)
                all_mask_weight.append(f_t)
            else:
                t = (conf[i] + conf[i + 1]) / 2
                f_t = math.exp(-self.c * t)
                all_mask_weight.append(f_t)

        # Judging each time interval in the time_matrix is at (1,8)(8,20)... Which interval in
        for i in range(len(conf)):
            if i == len(conf) - 1:
                mask = torch.where(time_matrix > conf[i], 1, 0)
            else:
                mask1 = torch.where(time_matrix > conf[i], 1, 0)
                mask2 = torch.where(time_matrix < conf[i + 1], 1, 0)
                mask = torch.mul(mask1, mask2)
            all_mask.append(mask)
        for i in range(len(all_mask)):
            time_f_w_matrix = time_f_w_matrix + ones * all_mask[i] * all_mask_weight[i]  # [B,L,L]

        return time_f_w_matrix

    def calculate_loss(self, interaction):

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        time_seq = interaction[self.timestamp]
        time_f_w_matrix = self.get_time_matrix(time_seq)
        seq_output = self.forward(item_seq, item_seq_len, time_f_w_matrix)

        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        time_seq = interaction[self.timestamp]
        time_f_w_matrix = self.get_time_matrix(time_seq)

        seq_output = self.forward(item_seq, item_seq_len, time_f_w_matrix)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores


    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

class MFTransformerEncoder(nn.Module):
    def __init__(self,
                 n_layers=2,
                 n_heads=2,
                 k_interests=5,
                 hidden_size=64,
                 seq_len=50,
                 inner_size=256,
                 hidden_dropout_prob=0.5,
                 attn_dropout_prob=0.5,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12):

        super(MFTransformerEncoder, self).__init__()
        layer = MFTransformerLayer(n_layers, n_heads, k_interests, hidden_size, seq_len, inner_size,
                                 hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(n_layers)])

    def forward(self,time_f_w_matrix, hidden_states, item_mask,position_embedding,output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TrandformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer layers' output,
            otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(time_f_w_matrix,hidden_states, item_mask,position_embedding)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class MFTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): the output of the point-wise feed-forward sublayer, is the output of the transformer layer
    """
    def __init__(self, n_layers,n_heads, k_interests, hidden_size, seq_len, intermediate_size,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps):
        super(MFTransformerLayer, self).__init__()
        self.multi_head_attention =MFMultiHeadAttention(n_layers,n_heads, k_interests, hidden_size,
                                       seq_len, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size,
                                         hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self,time_f_w_matrix, hidden_states, item_mask,position_embedding):
        attention_output,_,_ = self.multi_head_attention(time_f_w_matrix, hidden_states, item_mask,position_embedding)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class MFMultiHeadAttention(nn.Module):
    def __init__(self, n_layers,n_heads, k_interests, hidden_size, seq_len, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads))

        self.num_attention_heads = n_heads
        self.n_layers= n_layers
        self.hidden_size=hidden_size
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.k_interests = k_interests
        self.seq_len = seq_len
        # initialization for low-rank decomposed self-attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.attpooling_key = MaskedItemToInterestAggregation(hidden_size, self.n_layers, k_interests)
        # initialization for decoupled position encoding
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x): # transfor to multihead
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self,time_f_w_matrix,input_tensor,item_mask,pos_emb):                    #input_tensor:item_emb
        # linear map
        # input_tensor:[B,l,d]; item_mask:[B,l,d];time_f_w_matrix:[B,l,l]
        mixed_query_layer = self.query(input_tensor)     #[B,l,d] -> [B,l,d]
        mixed_key_layer = self.key(input_tensor)      #[B,l,d] -> [B,l,d]
        mixed_value_layer = self.value(input_tensor)        #[B,l,d] -> [B,l,d]
        # low-rank decomposed self-attention: relation of items
        query_layer = self.transpose_for_scores(
            mixed_query_layer).permute(0, 2, 1, 3)  # [B,l,d]  -> [B,h,l,h_d] ([B,n_heads,L,D/n_heads])h=number of heads, h_d=dim of each head
        key_layer = self.transpose_for_scores(
            self.attpooling_key(mixed_key_layer, item_mask,
                                time_f_w_matrix)).permute(0, 2, 3, 1)   # self.attpooling_key : [B,l,d]--->[B，k, d]
        #  [B,l,d]  -> key_layer: [B,h,k,h_d]
        #  [B,l,d]  -> value_layer: [B,h,k,h_d]
        value_layer = key_layer.permute(0, 1, 3, 2)

        #Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)     #[B,h,l,h_d]*[B,h,k,h_d].T = [B,h,l,k]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)        #attention_scores=q*k/{sqrt(d/h)}

        # normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)      #[B,h,l,k] * [B,h,k,h_d] = [B,h,l,h_d]

        # decoupled position encoding: relation of positions
        value_layer_pos = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)    #[B,l,d]--->[B,h,l,h_d]
        pos_emb = self.pos_ln(pos_emb).unsqueeze(0)
        pos_query_layer = self.transpose_for_scores(self.pos_q_linear(pos_emb)).permute(0, 2, 1, 3)* self.pos_scaling
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb)).permute(0, 2, 3, 1)

        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer)
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        abs_pos_bias = nn.Softmax(dim=-2)(abs_pos_bias)
        context_layer_pos = torch.matmul(abs_pos_bias, value_layer_pos)

        context_layer = context_layer_item + context_layer_pos

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  #[B,h,l,h_d] ---> [B,l,h,h_d]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  #[B,l,h_d]
        context_layer = context_layer.view(*new_context_layer_shape)     #[B,l,h,h_d] -->[B,l,h_d]

        context_layer = torch.where(item_mask, context_layer, torch.zeros_like(context_layer))
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states,item_mask,time_f_w_matrix


class MaskedItemToInterestAggregation(nn.Module):
    def __init__(self,  hidden_size,num_layers=3, k_interests=5):
        super().__init__()
        self.k_interests = k_interests  # k latent interests
        module_list = []
        input_dim = hidden_size
        for i in range(num_layers - 1):
            module_list.append(nn.Linear(input_dim, hidden_size))
            module_list.append(nn.ReLU())
            input_dim = hidden_size
            # module_list.append(nn.Dropout(p=drop_out))
        module_list.append(nn.Linear(hidden_size, k_interests, bias=False))
        self.MLP = nn.Sequential(*module_list)

    def forward(self, input_tensor,item_mask,time_f_w_matrix):  # [B, l, d] -> [B, k, d]
        # D_matrix = torch.matmul(input_tensor, self.theta)  # [B, l, k]
        D_matrix= self.MLP(input_tensor)                    #[B, l, d] -> [B, l, k]
        D_matrix = nn.Softmax(dim=-1)(D_matrix)
        # Masked Softmax
        item_mask=item_mask[:,:,0].unsqueeze(-1).repeat(1,1,self.k_interests)      # [B,l,d] ->[B, l, k]
        D_matrix = torch.where(item_mask,D_matrix,-10000*torch.ones_like(D_matrix))
        D_matrix = D_matrix / math.sqrt(D_matrix.shape[-1])
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        D_matrix = time_f_w_matrix[:, :, 0].unsqueeze(-1) * D_matrix  # time_f_w_matrix[:,:,0]Take the most recent moment
        #time_f_w_matrix:[B,l,l] --> [B,l,1]   [B,l,1] * [B,l,k] ---> [B,l,k]
        result = torch.einsum('nij, nik -> nkj', input_tensor, D_matrix)   # [B,k,l] * [B,l,d] = [B, k, d]
        #  D_matrix * input_tensor
        return result
