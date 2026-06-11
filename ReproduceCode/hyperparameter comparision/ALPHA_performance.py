from typing import Counter
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import torch.nn.functional as F
from tqdm import tqdm

from model import AnomalyGFM, LossFunction
from utils import load_train_mat, load_test_mat, preprocess_features, csr_to_edge_index, normalize_adj
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# CẤU HÌNH
DATASET_TRAIN = "Facebook_svd"
DATASET_TEST = ["reddit_svd", "Amazon_upu_svd", "Disney_svd", "amazon_svd", "yelp_svd", "questions_svd", "tolokers_svd", "elliptic_svd", "t_finance_svd"]

LR = 1e-4
EMBEDDING_DIM = 300
EPOCHS = 300
SEED = 0
DEVICE = torch.device('cpu')

# CỐ ĐỊNH SEED
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adj_train, feat_train, ano_labels_train, ano_labels_val, idx_train, idx_val = load_train_mat(DATASET_TRAIN)

feat_train = preprocess_features(feat_train)
edge_index_train = csr_to_edge_index(adj_train).to(DEVICE)

feat_train = torch.FloatTensor(np.array(feat_train)).to(DEVICE)
ano_labels_train = torch.FloatTensor(ano_labels_train).to(DEVICE)

adj_train = normalize_adj(adj_train) + sp.eye(adj_train.shape[0])
adj_train = torch.FloatTensor(adj_train.todense()).to(DEVICE)

records = []
for ALPHA in [0.6, 0.8, 1.0, 1.2]:
    print("="*30)
    print(f'Training with ALPHA: {ALPHA}')
    model = AnomalyGFM(input_dim=feat_train.shape[1],
                hidden_dim=EMBEDDING_DIM,
                output_dim=EMBEDDING_DIM,
                activation='prelu').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = LossFunction(alpha=ALPHA)

    for epoch in tqdm(range(EPOCHS+1)):
        model.train()
        optimizer.zero_grad()
    
        normal_prompt_raw = torch.randn(EMBEDDING_DIM).to(DEVICE)
        abnormal_prompt_raw = torch.randn(EMBEDDING_DIM).to(DEVICE)

        h, residual, logits, p_n, p_a = model(feat_train, edge_index_train, normal_prompt_raw, abnormal_prompt_raw, adj_train)

        logits_train = logits[idx_train].squeeze()
        residual_train = residual[idx_train]
        labels_train = ano_labels_train

        loss = loss_fn(logits_train, labels_train, residual_train,
                    p_n, p_a)

        loss.backward()
        optimizer.step()

    for dts_test in DATASET_TEST:
        if dts_test in ['elliptic_svd', 't_finance_svd']:
            BETA = 4.0
        else:
            BETA = 0.0

        # LOAD DATA
        adj_test, feat_test, ano_labels_test= load_test_mat(dts_test)

        print(f"\nTest dataset: {dts_test}")
        print(f'Nb test nodes: {ano_labels_test.shape[0]}. Including {Counter(np.squeeze(ano_labels_test))}')    
        print(f'Nb test edges: {adj_test.nnz // 2}')

        # PREPROCESS FEATURES
        feat_test = preprocess_features(feat_test)

        # EDGE INDEX (không chứa self-loops)
        edge_index_test = csr_to_edge_index(adj_test).to(DEVICE)

        # CONVERT TO TENSORS
        feat_test = torch.FloatTensor(np.array(feat_test)).to(DEVICE)

        # CHUẨN HÓA ADJ + THÊM SELF-LOOPS: A = A_norm + I >< GCN truyền thống A = (A + I)_norm
        adj_test = normalize_adj(adj_test) + sp.eye(adj_test.shape[0])    
        adj_test = torch.FloatTensor(adj_test.todense()).to(DEVICE)

        model.eval()
        with torch.no_grad():
            normal_prompt_raw = torch.randn(EMBEDDING_DIM).to(DEVICE)
            abnormal_prompt_raw = torch.randn(EMBEDDING_DIM).to(DEVICE)

            _, residual_test, _, p_n, p_a = model(feat_test, edge_index_test, normal_prompt_raw, abnormal_prompt_raw, adj_test, sparse = False)

            emb_residual_test = residual_test
            emb_residual_test = F.normalize(emb_residual_test, p=2, dim=1)
            p_a_test = F.normalize(p_a, p=2, dim=0)
            p_n_test = F.normalize(p_n, p=2, dim=0)
            
            score_abnormal = torch.matmul(emb_residual_test, p_a_test)
            score_normal = torch.matmul(emb_residual_test, p_n_test)

            ano_score_a = torch.exp(score_abnormal)
            ano_score_a = ano_score_a.cpu().numpy()

            ano_score_n = torch.exp(-score_normal)
            ano_score_n = ano_score_n.cpu().numpy()

            anomaly_score = ano_score_a + BETA * ano_score_n
            auc_measure = roc_auc_score(ano_labels_test, anomaly_score)
            AP_measure = average_precision_score(ano_labels_test, anomaly_score)
            print(f'AUROC:{auc_measure:.4f}')
            print(f'AUPRC: {AP_measure}')

            records.append({
                'dataset': dts_test,
                'alpha': ALPHA,
                'auc': auc_measure,
                'ap': AP_measure
            })

df = pd.DataFrame(records)
df.to_csv(str(SCRIPT_DIR / "ALPHA_results.csv"), index=False)
print("\nResults saved to ALPHA_results.csv")
