import torch
import copy
import torch.nn as nn
from model.layers_t7 import Embedding, VisualProjection, FeatureEncoder, CQAttention, CQConcatenate, \
    ConditionedPredictor, HighLightLayer
from transformers import AdamW, get_linear_schedule_with_warmup
from model.lgte_local import LocalTemporalEncoder
from model.lgte_global import GlobalTemporalEncoder
from model.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding
from model.model_components import MILNCELoss
from easydict import EasyDict as edict

def build_optimizer_and_scheduler(model, configs):
    no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no decay for parameters of layer norm and bias
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=configs.init_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, configs.num_train_steps * configs.warmup_proportion,
                                                configs.num_train_steps)
    return optimizer, scheduler


class TCFN(nn.Module):
    def __init__(self, configs, word_vectors):
        super(TCFN, self).__init__()
        self.configs = configs
        self.embedding_net = Embedding(num_words=configs.word_size, num_chars=configs.char_size, out_dim=configs.dim,
                                       word_dim=configs.word_dim, char_dim=configs.char_dim, word_vectors=word_vectors,
                                       drop_rate=configs.drop_rate)
        # self.video_affine = VisualProjection(visual_dim=configs.video_feature_dim, dim=configs.dim,
        #                                      drop_rate=configs.drop_rate)
        # self.feature_encoder = FeatureEncoder(dim=configs.dim, num_heads=configs.num_heads, kernel_size=7, num_layers=4,
        #                                       max_pos_len=configs.max_pos_len, drop_rate=configs.drop_rate)
        # video and query fusion
        # self.cq_attention = CQAttention(dim=configs.dim, drop_rate=configs.drop_rate)
        # self.cq_concat = CQConcatenate(dim=configs.dim)
        # query-guided highlighting
        self.highlight_layer = HighLightLayer(dim=configs.dim)
        # conditioned predictor
        self.predictor = ConditionedPredictor(dim=configs.dim, num_heads=configs.num_heads, drop_rate=configs.drop_rate,
                                              max_pos_len=configs.max_pos_len, predictor=configs.predictor)
        # init parameters
        self.init_parameters()

        # local-global encoder
        # print(configs.video_feature_dim)
        # print(configs.dim)
        self.hidden_dim_1d = self.configs.dim
        self.temporal_scale = configs.max_pos_len
        self.window_l = 16
        self.lgte_l_num = 1
        self.lgte_local = nn.ModuleList(
            [LocalTemporalEncoder(self.hidden_dim_1d, 0.1, self.temporal_scale, self.window_l) for idx_lgte in range(self.lgte_l_num)])
        self.lgte_local_1 = copy.deepcopy(self.lgte_local)
        self.lgte_local_2 = copy.deepcopy(self.lgte_local)
        self.lgte_local_3 = copy.deepcopy(self.lgte_local)
        self.lgte_local_p = copy.deepcopy(self.lgte_local)

        self.window_g = 16
        self.lgte_g_num = 1
        self.lgte_global = nn.ModuleList(
            [GlobalTemporalEncoder(self.hidden_dim_1d, 0.1, self.temporal_scale, self.window_g) for idx_lgte in range(self.lgte_g_num)])
        self.lgte_global_1 = copy.deepcopy(self.lgte_global)
        self.lgte_global_2 = copy.deepcopy(self.lgte_global)
        self.lgte_global_3 = copy.deepcopy(self.lgte_global)
        self.lgte_global_p = copy.deepcopy(self.lgte_global)

        # cross encoder
        self.video_input_proj = LinearLayer(configs.video_feature_dim, configs.dim, layer_norm=True,
                                            dropout=0.2, relu=True)
        self.video_encoder1 = BertAttention(edict(hidden_size=configs.dim, intermediate_size=configs.dim,
                                                  hidden_dropout_prob=configs.drop_rate, num_attention_heads=8,
                                                  attention_probs_dropout_prob=configs.drop_rate))
        # self.video_encoder2 = copy.deepcopy(self.video_encoder1)
        # self.video_encoder3 = copy.deepcopy(self.video_encoder1)
        self.ctx_pos_embed = TrainablePositionalEncoding(max_position_embeddings=configs.max_pos_len,
                                                         hidden_size=configs.dim, dropout=configs.drop_rate)
        # self.q_input_proj = LinearLayer(configs.dim, configs.dim, layer_norm=True,
        #                                 dropout=configs.drop_rate, relu=True)
        self.q_encoder1 = copy.deepcopy(self.video_encoder1)
        # self.q_encoder2 = copy.deepcopy(self.video_encoder1)
        # self.q_encoder3 = copy.deepcopy(self.video_encoder1)
        cross_att_cfg = edict(hidden_size=configs.dim, num_attention_heads=8,
                              attention_probs_dropout_prob=0.1)

        self.video_cross_att = BertSelfAttention(cross_att_cfg)

        self.video_cross_att_l_1 = copy.deepcopy(self.video_cross_att)
        self.video_cross_att_l_2 = copy.deepcopy(self.video_cross_att)
        self.video_cross_att_l_3 = copy.deepcopy(self.video_cross_att)
        self.video_cross_att_l_4 = copy.deepcopy(self.video_cross_att)

        self.video_cross_att_g_1 = copy.deepcopy(self.video_cross_att)
        self.video_cross_att_g_2 = copy.deepcopy(self.video_cross_att)
        self.video_cross_att_g_3 = copy.deepcopy(self.video_cross_att)
        self.video_cross_att_g_4 = copy.deepcopy(self.video_cross_att)

        self.video_cross_layernorm = nn.LayerNorm(configs.dim)

        self.video_cross_layernorm_l_1 = copy.deepcopy(self.video_cross_layernorm)
        self.video_cross_layernorm_l_2 = copy.deepcopy(self.video_cross_layernorm)
        self.video_cross_layernorm_l_3 = copy.deepcopy(self.video_cross_layernorm)
        self.video_cross_layernorm_l_4 = copy.deepcopy(self.video_cross_layernorm)

        self.video_cross_layernorm_g_1 = copy.deepcopy(self.video_cross_layernorm)
        self.video_cross_layernorm_g_2 = copy.deepcopy(self.video_cross_layernorm)
        self.video_cross_layernorm_g_3 = copy.deepcopy(self.video_cross_layernorm)
        self.video_cross_layernorm_g_4 = copy.deepcopy(self.video_cross_layernorm)

        self.q_cross_att = BertSelfAttention(cross_att_cfg)

        # self.q_cross_att_l_1 = copy.deepcopy(self.q_cross_att)
        # self.q_cross_att_l_2 = copy.deepcopy(self.q_cross_att)
        # self.q_cross_att_l_3 = copy.deepcopy(self.q_cross_att)
        #
        # self.q_cross_att_g_1 = copy.deepcopy(self.q_cross_att)
        # self.q_cross_att_g_2 = copy.deepcopy(self.q_cross_att)
        # self.q_cross_att_g_3 = copy.deepcopy(self.q_cross_att)

        self.q_cross_layernorm = nn.LayerNorm(configs.dim)

        # self.q_cross_layernorm_l_1 = copy.deepcopy(self.q_cross_layernorm)
        # self.q_cross_layernorm_l_2 = copy.deepcopy(self.q_cross_layernorm)
        # self.q_cross_layernorm_l_3 = copy.deepcopy(self.q_cross_layernorm)
        #
        # self.q_cross_layernorm_g_1 = copy.deepcopy(self.q_cross_layernorm)
        # self.q_cross_layernorm_g_2 = copy.deepcopy(self.q_cross_layernorm)
        # self.q_cross_layernorm_g_3 = copy.deepcopy(self.q_cross_layernorm)

        # self.query_encoder = BertAttention(edict(hidden_size=configs.dim, intermediate_size=configs.dim,
        #                                          hidden_dropout_prob=0.1, num_attention_heads=8,
        #                                          attention_probs_dropout_prob=0.1))
        # self.video_encoder2 = copy.deepcopy(self.query_encoder)

        # self.local_cross =

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()
        self.apply(init_weights)

    @staticmethod
    def cross_context_encoder(main_context_feat, main_context_mask, side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer: cross attention layer
            norm_layer: layer norm layer
        """
        cross_mask = torch.einsum("bm,bn->bmn", main_context_mask, side_context_mask)  # (N, Lq, Lk)
        # print(cross_mask)
        cross_out = cross_att_layer(main_context_feat, side_context_feat, side_context_feat, cross_mask)  # (N, Lq, D)
        residual_out = norm_layer(cross_out + main_context_feat)
        return residual_out

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def forward(self, word_ids, char_ids, video_features, v_mask, q_mask):

        # video_features = self.video_affine(video_features)

        # video_features = self.feature_encoder(video_features, mask=v_mask)

        query_features = self.embedding_net(word_ids, char_ids)
        query_features = self.q_encoder1(query_features, q_mask.unsqueeze(1))
        # query_features = self.feature_encoder(query_features, mask=q_mask)
        # print(v_mask.shape)


        # encoding video and subtitle features, respectively
        video_features = self.encode_input(video_features, v_mask, self.video_input_proj, self.video_encoder1,
                                           self.ctx_pos_embed)
        # query_features = self.encode_input(query_features, q_mask, self.q_input_proj, self.q_encoder1,
        #                                    self.ctx_pos_embed)
        # video_features = self.video_encoder1(video_features, v_mask.unsqueeze(1))

        # 1st cross attention
        # cross encoding video features
        q_cross = self.cross_context_encoder(query_features, q_mask, video_features, v_mask, self.q_cross_att, self.q_cross_layernorm)  # (N, L, D)
        # x_encoded_q_feat_ = self.q_encoder2(x_encoded_q_feat, q_mask.unsqueeze(1))
        # cross encoding query features
        v_cross = self.cross_context_encoder(video_features, v_mask, query_features, q_mask, self.video_cross_att, self.video_cross_layernorm)  # (N, L, D)
        # x_encoded_video_feat_ = self.video_encoder2(x_encoded_video_feat, v_mask.unsqueeze(1))

        # CAQ fusion module
        # fusion_features = self.cq_attention(x_encoded_video_feat_, x_encoded_q_feat_, v_mask, q_mask)

        # TFM Sequence
        # 1st l-g cross module
        for lgte_l_layer in self.lgte_local:
            l_features = lgte_l_layer(v_cross)
        for lgte_g_layer in self.lgte_global:
            g_features = lgte_g_layer(v_cross)
        g_features = g_features + l_features
        l_features = self.cross_context_encoder(l_features, v_mask, q_cross, q_mask, self.video_cross_att_l_1, self.video_cross_layernorm_l_1)
        g_features = self.cross_context_encoder(g_features, v_mask, q_cross, q_mask, self.video_cross_att_g_1, self.video_cross_layernorm_g_1)
        # g_features = g_features + l_features
        # l_q = self.cross_context_encoder(x_encoded_q_feat, q_mask, local_features, v_mask, self.q_cross_att_l_1, self.q_cross_layernorm_l_1)
        # g_q = self.cross_context_encoder(x_encoded_q_feat, q_mask, global_features, v_mask, self.q_cross_att_g_1, self.q_cross_layernorm_g_1)

        # 2nd l-g cross module
        for lgte_l_layer in self.lgte_local_1:
            l_features = lgte_l_layer(l_features)
        for lgte_g_layer in self.lgte_global_1:
            g_features = lgte_g_layer(g_features)
        g_features = g_features + l_features
        l_features = self.cross_context_encoder(l_features, v_mask, q_cross, q_mask, self.video_cross_att_l_2, self.video_cross_layernorm_l_2)
        g_features = self.cross_context_encoder(g_features, v_mask, q_cross, q_mask, self.video_cross_att_g_2, self.video_cross_layernorm_g_2)
        # g_features = g_features + l_features
        # l_q = self.cross_context_encoder(l_q, q_mask, l_features, v_mask, self.q_cross_att_l_2, self.q_cross_layernorm_l_2)
        # g_q = self.cross_context_encoder(g_q, q_mask, g_features, v_mask, self.q_cross_att_g_2, self.q_cross_layernorm_g_2)

        # 3rd l-g cross module
        for lgte_l_layer in self.lgte_local_2:
            l_features = lgte_l_layer(l_features)
        for lgte_g_layer in self.lgte_global_2:
            g_features = lgte_g_layer(g_features)
        g_features = g_features + l_features
        l_features = self.cross_context_encoder(l_features, v_mask, q_cross, q_mask, self.video_cross_att_l_3, self.video_cross_layernorm_l_3)
        g_features = self.cross_context_encoder(g_features, v_mask, q_cross, q_mask, self.video_cross_att_g_3, self.video_cross_layernorm_g_3)
        # g_features = g_features + l_features

        # 4th l-g cross module
        # for lgte_l_layer in self.lgte_local_3:
        #     l_features = lgte_l_layer(l_features)
        # for lgte_g_layer in self.lgte_global_3:
        #     g_features = lgte_g_layer(g_features)
        # g_features = g_features + l_features
        # l_features = self.cross_context_encoder(l_features, v_mask, q_cross, q_mask, self.video_cross_att_l_4, self.video_cross_layernorm_l_4)
        # g_features = self.cross_context_encoder(g_features, v_mask, q_cross, q_mask, self.video_cross_att_g_4, self.video_cross_layernorm_g_4)
        # g_features = g_features + l_features

        # boundary pred

        for lgte_l_layer in self.lgte_local_3:
            l_features = lgte_l_layer(l_features)
        for lgte_g_layer in self.lgte_global_3:
            g_features = lgte_g_layer(g_features)
        g_features = g_features + l_features

        start_logits, end_logits = self.predictor(l_features, mask=v_mask)
        # b-g pred
        h_score = self.highlight_layer(g_features, v_mask)
        # h_score = 0

        return h_score, start_logits, end_logits

    def extract_index(self, start_logits, end_logits):
        return self.predictor.extract_index(start_logits=start_logits, end_logits=end_logits)

    def extract_index_5r(self, start_logits, end_logits):
        return self.predictor.extract_index_5r(start_logits=start_logits, end_logits=end_logits)

    def compute_highlight_loss(self, scores, labels, mask):
        return self.highlight_layer.compute_loss(scores=scores, labels=labels, mask=mask)

    def compute_loss(self, start_logits, end_logits, start_labels, end_labels):
        return self.predictor.compute_cross_entropy_loss(start_logits=start_logits, end_logits=end_logits,
                                                         start_labels=start_labels, end_labels=end_labels)
