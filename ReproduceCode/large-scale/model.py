import torch
import torch.nn as nn
import torch.nn.functional as F 

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        """
        seq : [B, S, in_ft]
        adj : [B, S, S]
        """
        # Linear: [B, S, in] -> [B, S, out]
        seq_fts = self.fc(seq)

        # Batched adjacency × features
        out = torch.bmm(adj, seq_fts)   # [B, S, out]

        if self.bias is not None:
            out = out + self.bias  # broadcast to [B, S, out]

        return self.act(out)  # representation

    
class Model(nn.Module):
    def __init__(self, n_in_1, n_in_2, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.fc_map = nn.Linear(n_in_1, n_in_2, bias=False)

        self.gcn1 = GCN(n_in_2, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)

        self.fc1 = nn.Linear(n_h, 1, bias=False)
        self.fc2 = nn.Linear(n_h, 1, bias=False)

        self.act = nn.ReLU()

        self.fc_normal_prompt = nn.Linear(n_h, n_h, bias=False)
        self.fc_abnormal_prompt = nn.Linear(n_h, n_h, bias=False)

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

    def forward(self, seq1, adj, raw_adj, normal_prompt, abnormal_prompt, sparse=False):
        """
        seq1: [B, S, D]
        adj:  [B, S, S]
        raw_adj: [S, S]  OR [B, S, S]
        """

        B, S, _ = seq1.shape
        device = seq1.device

        # Batched raw_adj
        if raw_adj.dim() == 2:
            raw_adj = raw_adj.unsqueeze(0).expand(B, -1, -1)   # [B, S, S]

        # Remove self loops
        I = torch.eye(S, device=device).unsqueeze(0)  # [1, S, S]
        raw_adj = raw_adj * (1 - I)

        # Normalize per graph
        col_sum = raw_adj.sum(dim=2, keepdim=True)          # [B, S, 1]
        adj_norm = raw_adj / col_sum                        # [B, S, S]
        adj_norm[torch.isnan(adj_norm)] = 0
        adj_norm[torch.isinf(adj_norm)] = 0

        # GCN layers (batched)
        h1 = self.gcn1(seq1, adj, sparse)                   # [B, S, H]
        emb = self.gcn2(h1, adj, sparse)                    # [B, S, H]

        # Prototypes
        normal_prompt = self.act(self.fc_normal_prompt(normal_prompt))     # [H]
        abnormal_prompt = self.act(self.fc_abnormal_prompt(abnormal_prompt))  # [H]

        # Residual computation
        emb_neighbors = torch.bmm(adj_norm, emb)            # [B, S, H]
        emb_residual = emb - emb_neighbors                  # [B, S, H]

        # logits
        logit = self.fc1(emb)                               # [B, S, 1]
        logit_residual = self.fc2(emb_residual)             # [B, S, 1]

        return logit, logit_residual, emb, emb_residual, normal_prompt, abnormal_prompt


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out
