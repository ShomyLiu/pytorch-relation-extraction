# -*- coding: utf-8 -*-

from .BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCNN_ONE(BasicModule):
    '''
    Zeng 2015 DS PCNN
    '''
    def __init__(self, opt):
        super(PCNN_ONE, self).__init__()

        self.opt = opt

        self.model_name = 'PCNN_ONE'

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)

        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2

        # for more filter size
        self.convs = nn.ModuleList([nn.Conv2d(1, self.opt.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.opt.filters])

        all_filter_num = self.opt.filters_num * len(self.opt.filters)

        if self.opt.use_pcnn:
            all_filter_num = all_filter_num * 3
            masks = torch.FloatTensor(([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            if self.opt.use_gpu:
                masks = masks.cuda()
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(masks)
            self.mask_embedding.weight.requires_grad = False

        self.linear = nn.Linear(all_filter_num, self.opt.rel_num)
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.init_model_weight()
        self.init_word_emb()

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def init_word_emb(self):

        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.opt.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v

        w2v = p_2norm(self.opt.w2v_path)
        p1_2v = p_2norm(self.opt.p1_2v_path)
        p2_2v = p_2norm(self.opt.p2_2v_path)

        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
            self.pos1_embs.weight.data.copy_(p1_2v.cuda())
            self.pos2_embs.weight.data.copy_(p2_2v.cuda())
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)

    def mask_piece_pooling(self, x, mask):
        '''
        refer: https://github.com/thunlp/OpenNRE
        A fast piecewise pooling using mask
        '''
        x = x.unsqueeze(-1).permute(0, 2, 1, -1)
        masks = self.mask_embedding(mask).unsqueeze(-2) * 100
        x = masks.float() + x
        x = torch.max(x, 1)[0] - torch.FloatTensor([100]).cuda()
        x = x.view(-1, x.size(1) * x.size(2))
        return x

    def piece_max_pooling(self, x, insPool):
        '''
        old version piecewise
        '''
        split_batch_x = torch.split(x, 1, 0)
        split_pool = torch.split(insPool, 1, 0)
        batch_res = []
        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()  # all_filter_num * max_len
            pool = split_pool[i].squeeze().data    # 2
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)          # all_filter_num * 1
            seg_2 = ins[:, pool[0]: pool[1]].max(1)[0].unsqueeze(1)  # all_filter_num * 1
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)    # 1 * 3all_filter_num
            batch_res.append(piece_max_pool)

        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.opt.filters_num
        return out

    def forward(self, x, train=False):

        insEnt, _, insX, insPFs, insPool, insMasks = x
        insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]

        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)
        x = x.unsqueeze(1)
        x = self.dropout(x)

        x = [conv(x).squeeze(3) for conv in self.convs]
        if self.opt.use_pcnn:
            x = [self.mask_piece_pooling(i, insMasks) for i in x]
            # x = [self.piece_max_pooling(i, insPool) for i in x]
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1).tanh()
        x = self.dropout(x)
        x = self.linear(x)

        return x
