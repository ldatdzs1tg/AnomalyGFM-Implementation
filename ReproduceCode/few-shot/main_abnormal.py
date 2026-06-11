from typing import Counter
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import torch.nn.functional as F

from model import abnormalFewshotModel
from utils import load_mat, preprocess_features, csr_to_edge_index, normalize_adj
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# CẤU HÌNH
DATASET_TRAIN = "Facebook_svd"      # Chỉ gọi load_mat
DATASET_TEST = "reddit_svd"     # Target dataset để prompt tune + test

HIDDEN_DIM = 300
OUTPUT_DIM = 300  # EMBEDDING DIM
LR = 1e-4
ALPHA = 1.0
BETA = 0.5 # Dựa vào global avg edge similarity của mỗi dataset
K_SHOT = 10  # Số lượng mẫu few-shot để fine-tune prompt
EPOCHS = 20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0

if DATASET_TEST in ['elliptic_svd',  't_finance_svd']: # sim. < 0.5
    BETA = 4.0

# CỐ ĐỊNH SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# LOAD DATA
# Dùng zero-shot setting có sẵn + fine-tune prompt trên dataset test
_, adj_test, _, feat_test, _, _, ano_labels_test, _, _ = load_mat(DATASET_TRAIN, DATASET_TEST)

# FEW-SHOT SETTING
all_idx_test = list(range(ano_labels_test.shape[0]))
all_idx_test_abnormal = list(np.where(ano_labels_test == 1)[0])

random.shuffle(all_idx_test)
random.shuffle(all_idx_test_abnormal)
# few-shot id for training
few_shot_train_id = all_idx_test_abnormal[:K_SHOT]
# remaining id for testing
few_shot_eval_id = list(set(all_idx_test) - set(few_shot_train_id))

print("Train dataset:", DATASET_TRAIN)
print("\n Few-shot Test dataset:", DATASET_TEST)
print(f'Nb nodes: {ano_labels_test.shape[0]}. Including {Counter(np.squeeze(ano_labels_test))}')    
print(f'Nb edges: {adj_test.nnz // 2}')
print(f"Tuning with {len(few_shot_train_id)} labeled Abnormal nodes.")
print(f"Evaluating on {len(few_shot_eval_id)} nodes.")

# PREPROCESS FEATURES
feat_test = preprocess_features(feat_test)

# EDGE INDEX (không chứa self-loops)
edge_index_test = csr_to_edge_index(adj_test).to(DEVICE)

# CONVERT TO TENSORS
feat_test = torch.FloatTensor(np.array(feat_test)).to(DEVICE)

# CHUẨN HÓA ADJ + THÊM SELF-LOOPS: A = A_norm + I >< GCN truyền thống A = (A + I)_norm
adj_test = normalize_adj(adj_test) + sp.eye(adj_test.shape[0])    
adj_test = torch.FloatTensor(adj_test.todense()).to(DEVICE)


# DEFINE MODEL
INPUT_DIM = feat_test.shape[1]   # Số đặc trưng đầu vào
model = abnormalFewshotModel(input_dim=INPUT_DIM,
                   hidden_dim=HIDDEN_DIM,
                   output_dim=OUTPUT_DIM,
                   activation='prelu').to(DEVICE)

# LOAD WEIGHTS PRE-TRAINED TỪ ZERO-SHOT MODEL
checkpoint = torch.load(SCRIPT_DIR / 'model_weights_300.pth', map_location=DEVICE)
model.load_state_dict(checkpoint, strict=False)

for param in model.parameters():
    param.requires_grad = False
# Chỉ fine-tune các tham số của prompt
model.prompt.adaptation.weight.requires_grad = True
model.prompt.adaptation.bias.requires_grad = True
model.prompt.global_emb.requires_grad = True
# Kiểm tra các tham số được fine-tune
for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")
# Optimizer chỉ chứa các tham số cần fine-tune
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    normal_prompt_raw = torch.randn(OUTPUT_DIM).to(DEVICE)
    abnormal_prompt_raw = torch.randn(OUTPUT_DIM).to(DEVICE)

    h, residual, logits, p_n, p_a = model(feat_test, edge_index_test, normal_prompt_raw, abnormal_prompt_raw, adj_test)

    residual_few = residual[few_shot_train_id]

    diff_abnormal = torch.sqrt(torch.sum((residual_few - p_a) ** 2, dim=1))

    loss_pt = torch.mean(diff_abnormal)
    loss = loss_pt

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.4f}")

    if epoch % 2 == 0 or epoch == EPOCHS:
        print('<'*20 + ' TESTING ' + '>'*20)
        model.eval()
        with torch.no_grad():
            normal_prompt_raw = torch.randn(OUTPUT_DIM).to(DEVICE)
            abnormal_prompt_raw = torch.randn(OUTPUT_DIM).to(DEVICE)

            _, residual_test, _, p_n, p_a = model(feat_test, edge_index_test, normal_prompt_raw, abnormal_prompt_raw, adj_test)
            

            emb_residual_test = residual_test
            emb_residual_test = F.normalize(emb_residual_test, p=2, dim=1)
            p_a_test = F.normalize(p_a, p=2, dim=0).reshape(-1,1)
            p_n_test = F.normalize(p_n, p=2, dim=0).reshape(-1,1)
            
            score_abnormal = torch.matmul(emb_residual_test, p_a_test)
            score_normal = torch.matmul(emb_residual_test, p_n_test)
            ano_score_a = torch.exp(score_abnormal)
            ano_score_a = ano_score_a.cpu().numpy()

            ano_score_n = torch.exp(-score_normal)
            ano_score_n = ano_score_n.cpu().numpy()

            ano_score_mid = ano_score_a + BETA * ano_score_n   # BETA = 4 cho sim. <= 0.5
                                                            # BETA = 0 cho sim. > 0.5 -> Trùng với abnormal score
            auc_measure_mid = roc_auc_score(ano_labels_test[few_shot_eval_id], ano_score_mid[few_shot_eval_id])
            AP_measure_mid = average_precision_score(ano_labels_test[few_shot_eval_id]  , ano_score_mid[few_shot_eval_id])
            print(f'Testing {DATASET_TEST} AUROC:{auc_measure_mid:.4f}')
            print(f'Testing AUPRC: {AP_measure_mid}')

            print('<'*20 + ' TEST END ' + '>'*20)