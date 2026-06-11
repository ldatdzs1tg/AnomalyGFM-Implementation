import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_mean

class GCNLayer(nn.Module):               # Code GCN thủ công, từng lớp theo code trong src paper.
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCNLayer, self).__init__()
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
        # seq: (N, in_ft)
        # adj: (N, N) - dense hoặc sparse
        seq_fts = self.fc(seq)  # (N, out_ft)

        if sparse:
            # adj là sparse tensor COO
            out = torch.sparse.mm(adj, seq_fts)  # (N, out_ft)
        else:
            # adj là dense tensor
            out = torch.mm(adj, seq_fts)  # (N, out_ft)

        if self.bias is not None:
            out += self.bias

        return self.act(out)

class ResidualComputer(nn.Module):
    def __init__(self):
        super(ResidualComputer, self).__init__()
    def forward(self, h, edge_index):
        row, col = edge_index     # row = source, col = target  # edge_index không có self-loops
        neighbor_mean = scatter_mean(h[row], col, dim=0, dim_size=h.size(0))
        residual = h - neighbor_mean
        return residual
    
class SimplePrompt(nn.Module):
    def __init__(self, input_size):
        super(SimplePrompt, self).__init__()
        # global_emb đóng vai trò là learnable prompt
        self.global_emb = nn.Parameter(torch.Tensor(1, input_size))
        # a đóng vai trò là adapter layer
        self.adaptation = nn.Linear(input_size, input_size)
        self.act = nn.ReLU() 
        self.reset_parameters()

    def reset_parameters(self):
        # Khởi tạo Xavier
        torch.nn.init.xavier_uniform_(self.global_emb)
        torch.nn.init.xavier_uniform_(self.adaptation.weight)
        if self.adaptation.bias is not None:
            self.adaptation.bias.data.fill_(0.0)

    def forward(self, x):
        return x + self.act(self.adaptation(x)) + self.global_emb
    
class normalFewshotModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=300, activation='prelu'):
        super(normalFewshotModel, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, activation)
        self.gcn2 = GCNLayer(hidden_dim, output_dim, activation)
        self.res_computer = ResidualComputer()

        # Prototypes MLP
        self.fc_normal_prompt = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.fc_abnormal_prompt = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

        # Binary Classifier (MLP) đầu vào cho loss_bce
        self.classifier = nn.Linear(output_dim, 1) # Trả ra logits z để đưa vào BCEWithLogitsLoss

        #Few-shot Prompt 
        self.prompt = SimplePrompt(output_dim)

    def forward(self, features, edge_index, normal_prompt_raw, abnormal_prompt_raw, adj, sparse = False):
        h_1 = self.gcn1(features, adj, sparse)   
        h = self.gcn2(h_1, adj, sparse)       # GCN 2 lớp dùng adj = A norm + I >< GCN truyền thống dùng adj = (A + I) norm

        p_n = self.fc_normal_prompt(normal_prompt_raw)
        p_a = self.fc_abnormal_prompt(abnormal_prompt_raw)

        p_n = torch.squeeze(self.prompt(p_n))
        residual = self.res_computer(h, edge_index) # edge_index không có self-loops

        logits = self.classifier(h)
        return h, residual, logits, p_n, p_a

class abnormalFewshotModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=300, activation='prelu'):
        super(abnormalFewshotModel, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim, activation)
        self.gcn2 = GCNLayer(hidden_dim, output_dim, activation)
        self.res_computer = ResidualComputer()

        # Prototypes MLP
        self.fc_normal_prompt = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.fc_abnormal_prompt = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

        # Binary Classifier (MLP) đầu vào cho loss_bce
        self.classifier = nn.Linear(output_dim, 1) # Trả ra logits z để đưa vào BCEWithLogitsLoss

        #Few-shot Prompt 
        self.prompt = SimplePrompt(output_dim)

    def forward(self, features, edge_index, normal_prompt_raw, abnormal_prompt_raw, adj, sparse = False):
        h_1 = self.gcn1(features, adj, sparse)   
        h = self.gcn2(h_1, adj, sparse)       # GCN 2 lớp dùng adj = A norm + I >< GCN truyền thống dùng adj = (A + I) norm

        p_n = self.fc_normal_prompt(normal_prompt_raw)
        p_a = self.fc_abnormal_prompt(abnormal_prompt_raw)

        p_a = torch.squeeze(self.prompt(p_a))
        residual = self.res_computer(h, edge_index) # edge_index không có self-loops

        logits = self.classifier(h)
        return h, residual, logits, p_n, p_a
    
# class LossFunction(nn.Module):
#     def __init__(self, alpha = 1.0):
#         super(LossFunction, self).__init__()
#         self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
#         self.alpha = alpha

#     def forward(self, logits, labels, residual, p_n, p_a):
#         # BCE Loss
#         l_bce = self.bce_loss(logits, labels.float())

#         # Alignment Loss
#         res_n = residual[labels == 0]
#         res_a = residual[labels == 1]

#         diff_n = torch.sqrt(torch.sum((res_n - p_n)**2, dim=1)) if res_n.size(0) > 0 else torch.tensor(0.0)
#         diff_a = torch.sqrt(torch.sum((res_a - p_a)**2, dim=1)) if res_a.size(0) > 0 else torch.tensor(0.0)
    
#         l_align = torch.mean(diff_a) + 0.1*torch.mean(diff_n) # Phạt trọng số lớn hơn cho mẫu bất thường

#         # Tổng Loss
#         total_loss = l_bce + self.alpha * l_align

#         return total_loss
    