from typing import Counter
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import random
import torch.nn.functional as F

from model import AnomalyGFM, LossFunction
from utils import load_mat, preprocess_features, csr_to_edge_index, normalize_adj

# CẤU HÌNH
DATASET_TRAIN = "Facebook_svd"
DATASET_TEST = "Amazon_upu_svd"

HIDDEN_DIM = 300
OUTPUT_DIM = 300  # EMBEDDING DIM
LR = 1e-4
ALPHA = 1.0
BETA = 0.0 # Dựa vào global avg edge similarity của mỗi dataset
EPOCHS = 300
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0

# CỐ ĐỊNH SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# LOAD DATA
adj_train, adj_test, feat_train, feat_test, ano_labels_train, ano_labels_val, ano_labels_test, \
    idx_train, idx_val = load_mat(DATASET_TRAIN, DATASET_TEST)

print("\nTrain + Validation dataset:", DATASET_TRAIN)
print(f'Nb train nodes: {ano_labels_train.shape[0]}. Including {Counter(np.squeeze(ano_labels_train))}')
print(f'Nb validation nodes: {ano_labels_val.shape[0]}. Including {Counter(np.squeeze(ano_labels_val))}')
print(f'Nb train edges: {adj_train.nnz // 2}')

print("\nTest dataset:", DATASET_TEST)
print(f'Nb test nodes: {ano_labels_test.shape[0]}. Including {Counter(np.squeeze(ano_labels_test))}')    
print(f'Nb test edges: {adj_test.nnz // 2}')

# PREPROCESS FEATURES
feat_train = preprocess_features(feat_train)
feat_test = preprocess_features(feat_test)

# EDGE INDEX (không chứa self-loops)
edge_index_train = csr_to_edge_index(adj_train).to(DEVICE)  
edge_index_test = csr_to_edge_index(adj_test).to(DEVICE)

# CONVERT TO TENSORS
feat_train = torch.FloatTensor(np.array(feat_train)).to(DEVICE)
feat_test = torch.FloatTensor(np.array(feat_test)).to(DEVICE)

ano_labels_train = torch.FloatTensor(ano_labels_train).to(DEVICE)

# CHUẨN HÓA ADJ + THÊM SELF-LOOPS: A = A_norm + I >< GCN truyền thống A = (A + I)_norm
adj_train = normalize_adj(adj_train) + sp.eye(adj_train.shape[0])
adj_test = normalize_adj(adj_test) + sp.eye(adj_test.shape[0])    
adj_train = torch.FloatTensor(adj_train.todense()).to(DEVICE)
adj_test = torch.FloatTensor(adj_test.todense()).to(DEVICE)


# DEFINE MODEL
INPUT_DIM = feat_train.shape[1]
model = AnomalyGFM(input_dim=INPUT_DIM,
                   hidden_dim=HIDDEN_DIM,
                   output_dim=OUTPUT_DIM,
                   activation='prelu').to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

loss_fn = LossFunction(alpha=ALPHA)

for epoch in range(EPOCHS+1):
    model.train()
    optimizer.zero_grad()

    normal_prompt_raw = torch.randn(OUTPUT_DIM).to(DEVICE)
    abnormal_prompt_raw = torch.randn(OUTPUT_DIM).to(DEVICE)

    h, residual, logits, p_n, p_a = model(feat_train, edge_index_train, normal_prompt_raw, abnormal_prompt_raw, adj_train)

    logits_train = logits[idx_train].squeeze()
    residual_train = residual[idx_train]
    labels_train = ano_labels_train

    loss = loss_fn(logits_train, labels_train, residual_train,
                   p_n, p_a)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.4f}")

        logits_val = logits[idx_val].squeeze()
        logits_val = logits_val.detach().detach().cpu().numpy()

        auc = roc_auc_score(ano_labels_val, logits_val)
        AP = average_precision_score(ano_labels_val, logits_val, average='macro',
                                     pos_label=1, sample_weight=None)
        print(f'Validation Training {DATASET_TRAIN} AUROC: {auc:.4f}')
        print(f'Validation Training AUPRC: {AP}')

        emb_residual_val = residual[idx_val]
        emb_residual_val = F.normalize(emb_residual_val, p=2, dim=1)
        p_a_val = F.normalize(p_a, p=2, dim=0)
        p_n_val = F.normalize(p_n, p=2, dim=0)

        score_abnormal = torch.matmul(emb_residual_val, p_a_val)
        score_normal = torch.matmul(emb_residual_val, p_n_val)

        ano_score_a = torch.exp(score_abnormal)
        ano_score_a = ano_score_a.detach().cpu().numpy()
        auc_measure_abnormal = roc_auc_score(ano_labels_val, ano_score_a)
        AP_measure_abnormal = average_precision_score(ano_labels_val, ano_score_a)

        print(f'Val Abnormal  Training {DATASET_TRAIN} AUROC:{auc_measure_abnormal:.4f}')
        print(f'Val Abnormal Training AUPRC: {AP_measure_abnormal}')

        ano_score_n = torch.exp(-score_normal)
        ano_score_n = ano_score_n.detach().cpu().numpy()
        auc_measure_normal = roc_auc_score(ano_labels_val, ano_score_n)
        AP_measure_normal = average_precision_score(ano_labels_val, ano_score_n)

        print(f'Val Normal  Training {DATASET_TRAIN} AUROC:{auc_measure_normal:.4f}')
        print(f'Val Normal Training AUPRC: {AP_measure_normal}')

    if epoch % 50 == 0:
        print('<'*20 + ' TESTING ' + '>'*20)
        model.eval()
        with torch.no_grad():
            normal_prompt_raw = torch.randn(OUTPUT_DIM).to(DEVICE)
            abnormal_prompt_raw = torch.randn(OUTPUT_DIM).to(DEVICE)

            _, residual_test, logits_test, p_n, p_a = model(feat_test, edge_index_test, normal_prompt_raw, abnormal_prompt_raw, adj_test, sparse = False)

            logits_test = logits_test.squeeze()
            logits_test = logits_test.detach().cpu().numpy()
            auc = roc_auc_score(ano_labels_test, logits_test)
            AP = average_precision_score(ano_labels_test, logits_test, average='macro',
                                         pos_label=1, sample_weight=None)
            
            print(f'Epoch {epoch}/{EPOCHS}') 
            print(f'Testing {DATASET_TEST} AUROC: {auc:.4f}')
            print(f'Testing AUPRC: {AP}')

            print('--- Detailed Testing Scores ---')

            emb_residual_test = residual_test
            emb_residual_test = F.normalize(emb_residual_test, p=2, dim=1)
            p_a_test = F.normalize(p_a, p=2, dim=0)
            p_n_test = F.normalize(p_n, p=2, dim=0)
            
            score_abnormal = torch.matmul(emb_residual_test, p_a_test)
            score_normal = torch.matmul(emb_residual_test, p_n_test)

            ano_score_a = torch.exp(score_abnormal)
            ano_score_a = ano_score_a.cpu().numpy()
            auc_measure_abnormal = roc_auc_score(ano_labels_test, ano_score_a)
            AP_measure_abnormal = average_precision_score(ano_labels_test, ano_score_a)
            print(f'Abnormal Testing {DATASET_TEST} AUROC:{auc_measure_abnormal:.4f}')
            print(f'Abnormal Testing AUPRC: {AP_measure_abnormal}')
            print('-----')

            ano_score_n = torch.exp(-score_normal)
            ano_score_n = ano_score_n.cpu().numpy()
            auc_measure_normal = roc_auc_score(ano_labels_test, ano_score_n)
            AP_measure_normal = average_precision_score(ano_labels_test, ano_score_n)
            print(f'Normal Testing {DATASET_TEST} AUROC:{auc_measure_normal:.4f}')
            print(f'Normal Testing AUPRC: {AP_measure_normal}')

            print('-----')

            ano_score_mid = ano_score_a + 4 * ano_score_n   # BETA = 4 cho sim. <= 0.5
                                                            # BETA = 0 cho sim. > 0.5 -> Trùng với abnormal score
            auc_measure_mid = roc_auc_score(ano_labels_test, ano_score_mid)
            AP_measure_mid = average_precision_score(ano_labels_test, ano_score_mid)
            print(f'Mid Testing {DATASET_TEST} AUROC:{auc_measure_mid:.4f}')
            print(f'Mid Testing AUPRC: {AP_measure_mid}')

            print('<'*20 + ' TEST END ' + '>'*20)