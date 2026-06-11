import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import torch
from pathlib import Path

# Root of the repo: zero-shot/ -> ReproduceCode/ -> AnomalyGFM/
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets for AnomalyGFM"

def load_mat(dataset_train, dataset_test, train_rate=0.3, val_rate=0.1):

    data_train = sio.loadmat(str(DATASET_DIR / "{}.mat".format(dataset_train)))
    data_test = sio.loadmat(str(DATASET_DIR / "{}.mat".format(dataset_test)))

    label_train = data_train['Label'] if ('Label' in data_train) else data_train['gnd']
    attr_train = data_train['Attributes'] if ('Attributes' in data_train) else data_train['X']
    network_train = data_train['Network'] if ('Network' in data_train) else data_train['A']

    label_test = data_test['Label'] if ('Label' in data_test) else data_test['gnd']
    attr_test = data_test['Attributes'] if ('Attributes' in data_test) else data_test['X']
    network_test = data_test['Network'] if ('Network' in data_test) else data_test['A']

    adj_train = sp.csr_matrix(network_train)
    feat_train = sp.lil_matrix(attr_train)

    adj_test = sp.csr_matrix(network_test)
    feat_test = sp.lil_matrix(attr_test)

    all_labels = np.squeeze(np.array(label_train))
    ano_labels_test = np.squeeze(np.array(label_test))

    num_node_train = label_train.shape[1] # label_train: (1, N)
                                          # Hoặc adj_train: (N, N)
    num_train = int(num_node_train * train_rate)
    num_val = int(num_node_train * val_rate)
    all_idx_train = list(range(num_node_train))
    random.shuffle(all_idx_train)

    idx_train = all_idx_train[: num_train]
    idx_val = all_idx_train[num_train: num_train + num_val]

    ano_labels_train = all_labels[idx_train]
    ano_labels_val = all_labels[idx_val]

    return adj_train, adj_test, feat_train, feat_test, ano_labels_train, ano_labels_val, ano_labels_test, idx_train, idx_val

def preprocess_features(features):
    # Chuẩn hóa ma trận đặc trưng
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def csr_to_edge_index(adj):
    # Chuyển đổi csr_matrix sang edge index
    adj = adj.tocsr() # Đảm bảo adj ở dạng CSR
    row, col = adj.nonzero()
    edge_index = np.vstack((row, col))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def normalize_adj(adj):
    # Chuẩn hóa ma trận kề
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_adj

def load_train_mat(dataset, train_rate=0.3, val_rate=0.1):
    data = sio.loadmat(str(DATASET_DIR / "{}.mat".format(dataset)))

    label_train = data['Label'] if ('Label' in data) else data['gnd']
    attr_train = data['Attributes'] if ('Attributes' in data) else data['X']
    network_train = data['Network'] if ('Network' in data) else data['A']

    adj_train = sp.csr_matrix(network_train)
    feat_train = sp.lil_matrix(attr_train)

    all_labels = np.squeeze(np.array(label_train))

    num_node_train = label_train.shape[1] # label_train: (1, N)
                                          # Hoặc adj_train: (N, N)
    num_train = int(num_node_train * train_rate)
    num_val = int(num_node_train * val_rate)
    all_idx_train = list(range(num_node_train))
    random.shuffle(all_idx_train)

    idx_train = all_idx_train[: num_train]
    idx_val = all_idx_train[num_train: num_train + num_val]

    ano_labels_train = all_labels[idx_train]
    ano_labels_val = all_labels[idx_val]

    return adj_train, feat_train, ano_labels_train, ano_labels_val, idx_train, idx_val

def load_test_mat(dataset):
    data = sio.loadmat(str(DATASET_DIR / "{}.mat".format(dataset)))

    label_test = data['Label'] if ('Label' in data) else data['gnd']
    attr_test = data['Attributes'] if ('Attributes' in data) else data['X']
    network_test = data['Network'] if ('Network' in data) else data ['A']

    adj_test = sp.csr_matrix(network_test)
    feat_test = sp.lil_matrix(attr_test)

    ano_labels_test = np.squeeze(np.array(label_test))

    return adj_test, feat_test, ano_labels_test