import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import random
import torch.nn.functional as F
from tqdm import tqdm
from utils import normalize_adj, metrics_at_k
from pathlib import Path

SRC_DIR         = Path(__file__).resolve().parent          # b4e/src/
DATA_STRUCTURED = SRC_DIR.parent / "data_structured"       # b4e/data_structured/

EMBEDDING_DIM = 300
BETA = 4
SEED = 0
SUBGRAPH_SIZE = 4
DATASET = "b4e"

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
sample_feature = np.load(str(DATA_STRUCTURED / f'{DATASET}_feature_{SUBGRAPH_SIZE+1}_1.npy'))
sample_labels = np.load(str(DATA_STRUCTURED / f'{DATASET}_label_{SUBGRAPH_SIZE+1}_1.npy'))
sample_labels = sample_labels.squeeze()
sample_feature = torch.FloatTensor(sample_feature).to(DEVICE)

num_nodes = sample_labels.shape[0]

print("Dataset:", DATASET)
print("Nb of nodes:", num_nodes)
print("Subgraph size:", SUBGRAPH_SIZE)

print(sample_feature.shape)
print(sample_labels.shape)

# Load lại setting
model = torch.load(SRC_DIR / "model_residual2.pth", map_location=DEVICE)
model.eval()
print(model.__module__)
print(type(model))

# Dùng identity matrix -> Xử lý các node độc lập, không bị pha trộn thông tin với hàng xóm -> Trong 1 nhóm nhỏ thì detect anomaly tốt hơn
adj = torch.eye(SUBGRAPH_SIZE, SUBGRAPH_SIZE)
adj_norm = normalize_adj(adj)
adj_norm = torch.FloatTensor(adj_norm.todense()).to(DEVICE)
adj = torch.FloatTensor(adj).to(DEVICE)

# Prompts
normal_prompt_raw = torch.randn(EMBEDDING_DIM).to(DEVICE)
abnormal_prompt_raw = torch.randn(EMBEDDING_DIM).to(DEVICE)

# FAST INFERENCE
anomaly_scores = []

BATCH = 32768

with torch.no_grad():
    for start in tqdm(range(0, num_nodes, BATCH)):
        end = min(start + BATCH, num_nodes)

        # Lấy batch subgraphs
        feats = sample_feature[start:end]      # [B, S, D]

        # BATCHED ADJ       
        if end == num_nodes: # last batch có thể nhỏ hơn 1 batch
            adj_norm_batch = adj_norm.unsqueeze(0).repeat(end-start, 1, 1)
            adj_raw_batch  = adj.unsqueeze(0).repeat(end-start, 1, 1)
        else:                # batch
            adj_norm_batch = adj_norm.unsqueeze(0).repeat(BATCH, 1, 1)
            adj_raw_batch  = adj.unsqueeze(0).repeat(BATCH, 1, 1)


        _, _, _, emb_res, np_out, ap_out = model(
            feats,
            adj_norm_batch,
            adj_raw_batch,
            normal_prompt_raw,
            abnormal_prompt_raw
        )
        
        # Last node is the test node -> Compute last node residual embeddings
        emb_res = emb_res[:, -1, :]   # [B, H]
    
        # Normalize & compute scores in batch (nhanh)
        emb_norm = F.normalize(emb_res, p=2, dim=1)
        p_n = F.normalize(np_out.unsqueeze(0), p=2, dim=1)     # [1, H]
        p_a = F.normalize(ap_out.unsqueeze(0), p=2, dim=1)     # [1, H]

        score_ab = torch.exp(torch.sum(emb_norm * p_a, dim=1))
        score_no = torch.exp(-torch.sum(emb_norm * p_n, dim=1))

        anomaly_scores.extend((score_ab + BETA * score_no).cpu().numpy())

# Evaluate
ano_score = np.array(anomaly_scores)
auc = roc_auc_score(sample_labels, ano_score)
AP  = average_precision_score(sample_labels, ano_score)

print("Testing AUC: {:.4f}".format(auc))
print("Testing AP: ", AP)

precision_k, recall_k= metrics_at_k(sample_labels, ano_score, k_ratio=0.001)
print(f'Precision@0.1%: {precision_k:.4f}, Recall@0.1%: {recall_k:.4f}')

precision_k, recall_k= metrics_at_k(sample_labels, ano_score, k_ratio=0.0005)
print(f'Precision@0.05%: {precision_k:.4f}, Recall@0.05%: {recall_k:.4f}')