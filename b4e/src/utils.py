import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io import loadmat
import scipy.sparse as sp
import random
import torch
import dgl
from tqdm import tqdm
import functools
from collections import defaultdict
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

# src/ is here; navigate up one level to b4e/ root, then into subfolders
SRC_DIR        = Path(__file__).resolve().parent          # b4e/src/
B4E_DIR        = SRC_DIR.parent                           # b4e/
DATA_RAW       = B4E_DIR / "data_raw"                     # b4e/data_raw/
DATA_STRUCTURED = B4E_DIR / "data_structured"             # b4e/data_structured/
DATASET_DIR    = B4E_DIR.parent / "datasets for AnomalyGFM"  # benchmark datasets

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

def cmp_udf(x1, x2):           # key for sorting time from oldest to latest
    time1 = int(x1[2])
    time2 = int(x2[2])
    if time1 < time2:
        return -1
    elif time1 > time2:
        return 1
    else:
        return 0

def cmp_udf_reverse(x1, x2):  # latest to oldest
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0

def load_data(f_in, f_out):
    eoa2seq_out = {}
    error_trans = []
    while True:
        trans = f_out.readline()
        if trans == "":
            break
        record = trans.split(",")
        trans_hash = record[0]
        block_number = int(record[3])
        from_address = record[5]
        to_address = record[6]
        value = int(record[7]) / (pow(10, 12))
        gas = int(record[8])
        gas_price = int(record[9])
        block_timestamp = int(record[11])
        if from_address == "" or to_address == "":
            error_trans.append(trans)
            continue
        try:
            eoa2seq_out[from_address].append([to_address, block_number, block_timestamp, value, "OUT", 1])
        except:
            eoa2seq_out[from_address] = [[to_address, block_number, block_timestamp, value, "OUT", 1]]

    eoa2seq_in = {}
    while True:
        trans = f_in.readline()
        if trans == "":
            break
        record = trans.split(",")
        block_number = int(record[3])
        from_address = record[5]
        to_address = record[6]
        value = int(record[7]) / (pow(10, 12))
        gas = int(record[8])
        gas_price = int(record[9])
        block_timestamp = int(record[11])
        if from_address == "" or to_address == "":
            error_trans.append(trans)
            continue
        try:
            eoa2seq_in[to_address].append([from_address, block_number, block_timestamp, value, "IN", 1]) # not process trans
        except:
            eoa2seq_in[to_address] = [[from_address, block_number, block_timestamp, value, "IN", 1]] # in/out, cnt
    return eoa2seq_in, eoa2seq_out

def seq_duplicate(eoa2seq_in, eoa2seq_out):          # merge transactions within 3 days for same account pairs
    eoa2seq_agg_in = {}
    for eoa in eoa2seq_in.keys():
        if len(eoa2seq_in[eoa]) >= 10000:  # >10000 transactions, skip
            continue
        seq_sorted = sorted(eoa2seq_in[eoa], key=functools.cmp_to_key(cmp_udf)) # sort by time ascending
        seq_tmp = [e.copy() for e in seq_sorted]
        for i in range(len(seq_tmp) - 1, 0, -1):
            l_acc = seq_tmp[i][0]  # latter
            f_acc = seq_tmp[i - 1][0]  # former
            l_time = int(seq_tmp[i][2])
            f_time = int(seq_tmp[i - 1][2])
            delta_time = l_time - f_time
            if f_acc != l_acc or delta_time > 86400 * 3:   # >3 days -> no merge
                continue
            # value add
            seq_tmp[i - 1][3] += seq_tmp[i][3]
            seq_tmp[i - 1][5] += seq_tmp[i][5]
            del seq_tmp[i]
        eoa2seq_agg_in[eoa] = seq_tmp

    eoa2seq_agg_out = {}
    for eoa in eoa2seq_out.keys():
        if len(eoa2seq_out[eoa])>=10000: # >10000 transactions, skip
            continue
        seq_sorted = sorted(eoa2seq_out[eoa], key=functools.cmp_to_key(cmp_udf)) # sort by time ascending
        seq_tmp = [e.copy() for e in seq_sorted]
        for i in range(len(seq_tmp) - 1, 0, -1):   # ascend from latest 
            l_acc = seq_tmp[i][0]  # latter
            f_acc = seq_tmp[i - 1][0]  # former
            l_time = int(seq_tmp[i][2])
            f_time = int(seq_tmp[i - 1][2])
            delta_time = l_time - f_time
            if f_acc != l_acc or delta_time > 86400 * 3:  # >3 days -> no merge
                continue
            # value add
            seq_tmp[i - 1][3] += seq_tmp[i][3]
            seq_tmp[i - 1][5] += seq_tmp[i][5]
            del seq_tmp[i]
        eoa2seq_agg_out[eoa] = seq_tmp

    eoa_list = list(eoa2seq_agg_out.keys()) # eoa_list must include eoa account only (i.e., have out transaction at least)
    eoa2seq_agg = {}

    for eoa in eoa_list:
        out_seq = eoa2seq_agg_out[eoa]
        try:
            in_seq = eoa2seq_agg_in[eoa]
        except:
            in_seq = []

        seq_agg = sorted(out_seq + in_seq, key=functools.cmp_to_key(cmp_udf_reverse))
        cnt_all = 0
        for trans in seq_agg:
            cnt_all += trans[5]
            # if cnt_all >= 20 and cnt_all<=10000:
            if cnt_all > 2 and cnt_all<=10000:
                eoa2seq_agg[eoa] = seq_agg
                break

    return eoa2seq_agg
def seq_generation(eoa2seq_in, eoa2seq_out):  # no merge transactions

    eoa_list = list(eoa2seq_out.keys()) # eoa_list must include eoa account only (i.e., have out transaction at least)
    eoa2seq = {}
    for eoa in eoa_list:
        out_seq = eoa2seq_out[eoa]
        try:
            in_seq = eoa2seq_in[eoa]
        except:
            in_seq = []
        seq_agg = sorted(out_seq + in_seq, key=functools.cmp_to_key(cmp_udf_reverse))
        cnt_all = 0
        for trans in seq_agg:
            cnt_all += 1
            # if cnt_all >= 5 and cnt_all<=10000:
            if cnt_all > 2 and cnt_all<=10000:
                eoa2seq[eoa] = seq_agg
                break

    return eoa2seq

def extract_features_to_csv(eoa2seq_agg, output_filename):
    features_list = []

    for eoa, transactions in eoa2seq_agg.items():
        # Phân loại IN và OUT
        in_txs = [tx for tx in transactions if tx[4] == "IN"]
        out_txs = [tx for tx in transactions if tx[4] == "OUT"]
        
        # 1. Volume Features
        val_in = [tx[3] for tx in in_txs]
        val_out = [tx[3] for tx in out_txs]
        count_in = [tx[5] for tx in in_txs]
        count_out = [tx[5] for tx in out_txs]
        
        total_val_in = sum(val_in)
        total_val_out = sum(val_out)
        total_count_in = sum(count_in)
        total_count_out = sum(count_out)
        balance = total_val_in - total_val_out

        # Clusters
        total_clusters_in = len(in_txs)
        total_clusters_out = len(out_txs)
        
        # 2. Temporal Features
        timestamps = sorted([int(tx[2]) for tx in transactions])
        lifespan = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 3600   # 1h default -> avoid zero lifespan (only 1 transaction)
        
        time_diffs = np.diff(timestamps) if len(timestamps) > 1 else [0]
        avg_time_diff = np.mean(time_diffs)
        std_time_diff = np.std(time_diffs)
        
        # 3. Graph/Relational Features
        unique_in_addr = len(set([tx[0] for tx in in_txs]))
        unique_out_addr = len(set([tx[0] for tx in out_txs]))
        unique_addr = len(set([tx[0] for tx in transactions]))
        
        # 4. Aggregation Features
        # Tỷ lệ gộp: Càng cao nghĩa là ví này giao dịch rất thường xuyên với cùng 1 đối tác
        avg_agg_count_in = np.mean(count_in) if count_in else 0
        avg_agg_count_out = np.mean(count_out) if count_out else 0
        avg_agg_count_total = np.mean(count_in + count_out) if (count_in + count_out) else 0

        # Tổng hợp vào một dictionary
        feat = {
            'address': eoa,
            # Volume
            'total_val_in': total_val_in,
            'total_val_out': total_val_out,
            'avg_val_in': np.mean(val_in) if val_in else 0,                             # cluster IN value velocity
            'avg_val_out': np.mean(val_out) if val_out else 0,                          # cluster OUT value velocity
            'avg_tx_val_in': total_val_in / (total_count_in + 1e-9),                    # transaction IN value velocity
            'avg_tx_val_out': total_val_out / (total_count_out + 1e-9),                 # transaction OUT value velocity
            'balance': balance,
            'turnover_rate': min(total_val_in, total_val_out) / (max(total_val_in, total_val_out) + 1e-9),
            'net_flow_ratio': (total_val_in - total_val_out) / (total_val_in + total_val_out + 1e-9),
            
            # Count & Degree
            'total_tx_in': total_count_in,
            'total_tx_out': total_count_out,
            'unique_src_addresses': unique_in_addr,
            'unique_dst_addresses': unique_out_addr,
            'tx_per_address': (total_count_in + total_count_out) / (unique_addr + 1e-9),  # concentration factor
            'in_out_addr_ratio': unique_in_addr / (unique_out_addr + 1e-9),
            'addr_flow_ratio': (unique_in_addr - unique_out_addr) / (unique_in_addr + unique_out_addr + 1e-9),
            'addr_reuse_rate': 1 - (unique_addr / (total_count_in + total_count_out + 1e-9)),
            
            # Temporal
            'lifespan_days': lifespan / 86400,
            'avg_hours_between_tx': avg_time_diff / 3600,
            'std_hours_between_tx': std_time_diff / 3600,
            'tx_per_hour': (total_count_in + total_count_out) / (lifespan / 3600 + 1e-9),       
            'overall_val_velocity': (total_val_in + total_val_out) / (lifespan / 3600 + 1e-9),  # overall value velocity
            
            # Aggregation behavior
            'total_clusters_in': total_clusters_in,
            'total_clusters_out': total_clusters_out,
            'tx_per_clusters': avg_agg_count_total,
            'avg_clusters_in': avg_agg_count_in,
            'avg_clusters_out': avg_agg_count_out,
            'max_clusters_size': max(count_in + count_out) if (count_in + count_out) else 0
        }
        features_list.append(feat)

    # Chuyển thành DataFrame và lưu CSV
    df = pd.DataFrame(features_list)
    df.to_csv(output_filename, index=False)
    print(f"Đã trích xuất đặc trưng cho {len(df)} ví và lưu vào {output_filename}")
    return df

def generate_bitcoin_mat(eoa2seq_agg, phishing_path, output_filename="bitcoin_dataset.mat"):
    print(">>> BẮT ĐẦU XỬ LÝ DỮ LIỆU...")

    # --- 1. NODE DISCOVERY & MAPPING ---
    # Mục tiêu: Tìm tất cả các ví xuất hiện trong dữ liệu (Key + Partner)
    print("1. Đang quét toàn bộ node...")
    all_nodes = set(eoa2seq_agg.keys())
    for tx_list in eoa2seq_agg.values():
        for tx in tx_list:
            all_nodes.add(tx[0]) # tx[0] là địa chỉ đối tác
            
    # Sắp xếp và đánh index cho node
    node_list = sorted(list(all_nodes))
    node_map = {addr: i for i, addr in enumerate(node_list)}
    num_nodes = len(node_list)
    print(f"   => Tổng số node tìm thấy: {num_nodes}")

    # --- 2. ACCUMULATE STATS & BUILD NETWORK ---
    # Mục tiêu: Xây dựng đồ thị và gom dữ liệu thống kê cho TỪNG node
    print("2. Đang xây dựng Network và thống kê dữ liệu...")
    adj_mat = lil_matrix((num_nodes, num_nodes), dtype=np.int8)
    
    # Dict chứa list các giá trị thô để tính feature sau này
    # Key là index của node để tiết kiệm memory
    stats = defaultdict(lambda: {
        'val_in': [], 'val_out': [],        # List giá trị tiền
        'count_in': [], 'count_out': [],    # List số lượng tx (trong cụm)
        'ts': [],                           # List timestamp
        'partners_in': set(), 'partners_out': set() # List đối tác duy nhất
    })

    for key_addr, tx_list in eoa2seq_agg.items():
        if key_addr not in node_map: continue
        u = node_map[key_addr] # Node chủ
        
        for tx in tx_list:
            # Parse transaction: [partner, block, ts, value, direction, count]
            partner_addr = tx[0]
            ts = int(tx[2])
            val = float(tx[3])
            direction = tx[4]
            count = int(tx[5])
            
            if partner_addr not in node_map: continue
            v = node_map[partner_addr] # Node đối tác

            if direction == "IN": 
                # Partner (v) -> Key (u)
                adj_mat[v, u] = 1 # Cạnh v -> u
                
                # Cập nhật cho Key (u) - Vai trò: Nhận
                stats[u]['val_in'].append(val)
                stats[u]['count_in'].append(count)
                stats[u]['ts'].append(ts)
                stats[u]['partners_in'].add(v)
                
                # Cập nhật cho Partner (v) - Vai trò: Gửi (Suy luận ngược)
                stats[v]['val_out'].append(val)
                stats[v]['count_out'].append(count)
                stats[v]['ts'].append(ts)
                stats[v]['partners_out'].add(u)
                
            else: # OUT
                # Key (u) -> Partner (v)
                adj_mat[u, v] = 1 # Cạnh u -> v
                
                # Cập nhật cho Key (u) - Vai trò: Gửi
                stats[u]['val_out'].append(val)
                stats[u]['count_out'].append(count)
                stats[u]['ts'].append(ts)
                stats[u]['partners_out'].add(v)
                
                # Cập nhật cho Partner (v) - Vai trò: Nhận (Suy luận ngược)
                stats[v]['val_in'].append(val)
                stats[v]['count_in'].append(count)
                stats[v]['ts'].append(ts)
                stats[v]['partners_in'].add(u)

    print(f"   => Đã xây dựng xong Network với {adj_mat.nnz} cạnh.")

    # --- 3. FEATURE EXTRACTION (Dựa trên logic của bạn) ---
    print("3. Đang tính toán Feature Matrix...")
    raw_features = []
    
    # Thứ tự feature để tiện theo dõi log-transform
    # [0:VolumeIn, 1:VolumeOut, 2:AvgValIn, 3:AvgValOut, 4:AvgTxIn, 5:AvgTxOut, 6:Balance, 
    #  7:Turnover, 8:NetFlow, 9:TxIn, 10:TxOut, 11:UniqueSrc, 12:UniqueDst, 13:TxPerAddr, 
    #  14:InOutRatio, 15:AddrFlow, 16:AddrReuse, 17:Lifespan, 18:AvgTime, 19:StdTime, 
    #  20:TxPerHour, 21:OverallVel, 22:ClustIn, 23:ClustOut, 24:TxPerClust, 
    #  25:AvgClustIn, 26:AvgClustOut, 27:MaxClust]

    for i in range(num_nodes):
        s = stats[i]
        
        # --- Prepare Raw Data ---
        val_in = s['val_in']
        val_out = s['val_out']
        count_in = s['count_in']
        count_out = s['count_out']
        ts = sorted(s['ts'])
        
        # 1. Volume
        total_val_in = sum(val_in)
        total_val_out = sum(val_out)
        total_count_in = sum(count_in)
        total_count_out = sum(count_out)
        balance = total_val_in - total_val_out
        
        # 2. Clusters (Số lượng phần tử trong list chính là số cụm vì data đã gộp)
        total_clusters_in = len(val_in)
        total_clusters_out = len(val_out)
        
        # 3. Temporal
        if len(ts) > 1:
            lifespan = ts[-1] - ts[0]
            diffs = np.diff(ts)
            avg_time_diff = np.mean(diffs)
            std_time_diff = np.std(diffs)
        else:
            lifespan = 0
            avg_time_diff = 0
            std_time_diff = 0
            
        # 4. Graph/Relational
        unique_in_addr = len(s['partners_in'])
        unique_out_addr = len(s['partners_out'])
        unique_addr = unique_in_addr + unique_out_addr
        
        # 5. Aggregation
        avg_agg_count_in = np.mean(count_in) if count_in else 0
        avg_agg_count_out = np.mean(count_out) if count_out else 0
        all_counts = count_in + count_out
        avg_agg_count_total = np.mean(all_counts) if all_counts else 0
        max_clusters_size = max(all_counts) if all_counts else 0

        # --- Construct Feature Vector (Theo đúng danh sách của bạn) ---
        feat = [
            # Volume
            total_val_in,
            total_val_out,
            np.mean(val_in) if val_in else 0,   # avg_val_in
            np.mean(val_out) if val_out else 0, # avg_val_out
            total_val_in / (total_count_in + 1e-9),  # avg_tx_val_in
            total_val_out / (total_count_out + 1e-9), # avg_tx_val_out
            balance,
            min(total_val_in, total_val_out) / (max(total_val_in, total_val_out) + 1e-9), # turnover_rate
            (total_val_in - total_val_out) / (total_val_in + total_val_out + 1e-9),       # net_flow_ratio
            
            # Count & Degree
            total_count_in,
            total_count_out,
            unique_in_addr,
            unique_out_addr,
            (total_count_in + total_count_out) / (unique_addr + 1e-9), # tx_per_address
            unique_in_addr / (unique_out_addr + 1e-9),                 # in_out_addr_ratio
            (unique_in_addr - unique_out_addr) / (unique_in_addr + unique_out_addr + 1e-9), # addr_flow_ratio
            1 - (unique_addr / (total_count_in + total_count_out + 1e-9)),                  # addr_reuse_rate
            
            # Temporal
            lifespan / 86400,          # lifespan_days
            avg_time_diff / 3600,      # avg_hours
            std_time_diff / 3600,      # std_hours
            (total_count_in + total_count_out) / (lifespan / 3600 + 1e-9),      # tx_per_hour
            (total_val_in + total_val_out) / (lifespan / 3600 + 1e-9),          # overall_val_velocity
            
            # Aggregation
            total_clusters_in,
            total_clusters_out,
            avg_agg_count_total,       # tx_per_clusters
            avg_agg_count_in,
            avg_agg_count_out,
            max_clusters_size
        ]
        raw_features.append(feat)

    X = np.array(raw_features, dtype=np.float32)
    print(f"   => Raw Feature Matrix shape: {X.shape}")

    # --- 4. PREPROCESSING (LOG & SCALE) ---
    print("4. Preprocessing...")
    
    # Danh sách các cột cần Log Transform (những cột có giá trị lớn, lệch)
    # Tránh log các cột Ratio/Rate (vì nó đã nằm trong khoảng -1 đến 1 hoặc 0 đến 1)
    # Indices: 0,1,2,3,4,5 (Volume variants), 9,10,11,12 (Counts), 
    #          17,18,19 (Time), 20,21 (Velocity), 22,23,24,25,26,27 (Clusters)
    log_indices = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    
    # Xử lý Log1p cho các cột dương
    for col in log_indices:
        X[:, col] = np.log1p(X[:, col])
        
    # Xử lý riêng cột Balance (Index 6) vì có thể âm: sign(x) * log(1 + |x|)
    X[:, 6] = np.sign(X[:, 6]) * np.log1p(np.abs(X[:, 6]))
    
    # Xử lý NaN/Inf phát sinh
    X = np.nan_to_num(X)
    
    # Standard Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = np.clip(X_scaled, -5, 5)
    
    print("   => Data clipped to range [-5, 5]")

    # --- 5. LABEL GENERATION ---
    print("5. Đang map Label từ file txt...")
    labels = np.zeros((num_nodes, 1), dtype=int)
    try:
        with open(phishing_path, 'r') as f:
            phish_set = set(line.strip() for line in f)
        
        count_phish = 0
        for i, addr in enumerate(node_list):
            if addr in phish_set:
                labels[i] = 1
                count_phish += 1
        print(f"   => Tìm thấy {count_phish} ví Phishing trong dữ liệu.")
    except FileNotFoundError:
        print("   => Cảnh báo: Không tìm thấy file Label, toàn bộ nhãn là 0.")

    # --- 6. SAVE TO .MAT ---
    print(f"6. Lưu file {output_filename}...")
    
    # Network phải ở dạng CSC (Compressed Sparse Column)
    network_csc = csc_matrix(adj_mat)
    
    mat_data = {
        'Network': network_csc,       # Ma trận kề
        'Attributes': X_scaled,       # Ma trận đặc trưng (đã chuẩn hóa)
        'Label': labels,              # Nhãn (0/1)
        'node_ids': np.array(node_list, dtype=object) # Mapping ID để tra cứu
    }
    
    sio.savemat(output_filename, mat_data)
    print(">>> HOÀN TẤT! File .mat đã sẵn sàng.")

def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim # Đảm bảo số chiều ban đầu >= chiều mong muốn
    svd = TruncatedSVD(n_components=out_dim, n_iter=7, random_state=42)
    newdata = svd.fit_transform(data)
    return newdata

def generate_svd_mat(dataset_str='b4e', emb_dimension=10):
    data = loadmat(str(DATA_STRUCTURED / '{}.mat'.format(dataset_str)))

    label = data['Label'] if ('Label' in data) else data['gnd']          
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    adj = data['Network'] if ('Network' in data) else data['A'] 

    newFeat = x_svd(attr, emb_dimension)

    print('Original feature shape: ', attr.shape)
    print('Reduced feature shape (Unified): ', newFeat.shape)

    out_path = str(DATA_STRUCTURED / '{}_svd_{}.mat'.format(dataset_str, emb_dimension))
    sio.savemat(out_path, {'Network': adj, 'Label': label, 'Attributes': newFeat})
    print(f'Saved SVD mat to {out_path}')
    
def generate_npy_for_fast_inference(mat_path, output_name, subgraph_size=6):
    """
    mat_path: Đường dẫn file b4e_svd_10.mat
    output_name: Tên dataset (ví dụ: 'b4e')
    subgraph_size: S
    """
    print(f"Loading {mat_path}...")
    data = sio.loadmat(str(mat_path))
    adj = data['Network']
    features = data['Attributes']
    labels = data['Label'].squeeze()

    # --- BƯỚC MỚI: CHUẨN HÓA LẦN 2 (POST-SVD) ---
    print(">>> Performing Post-SVD Normalization...")
    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    
    # 1. Chuyển sang DGL Graph để sampling cực nhanh
    g = dgl.from_scipy(adj)
    num_nodes = features.shape[0]
    feat_dim = features.shape[1]
    
    print(f"Sampling for {num_nodes} nodes...")
    
    # 2. Random Walk lấy mẫu (Lấy bước đi dài hơn S để lọc trùng)
    # Chúng ta lấy độ dài = subgraph_size * 2
    traces, _ = dgl.sampling.random_walk(g, nodes=torch.arange(num_nodes), length=subgraph_size * 2)
    
    # 3. Xây dựng ma trận feature khổng lồ [N, S, D]
    # Khởi tạo mảng trắng trên RAM
    final_feats = np.zeros((num_nodes, subgraph_size, feat_dim), dtype=np.float32)
    
    for i in tqdm(range(num_nodes)):
        # Lấy các node hàng xóm từ bước đi, loại bỏ chính nó
        walk = traces[i]
        unique_nodes = torch.unique(walk, sorted=False)
        neighbors = unique_nodes[unique_nodes != i].tolist()
        
        # Padding / Truncating cho S-1 node đầu tiên
        needed = subgraph_size - 1
        if len(neighbors) > needed:
            neighbors = neighbors[:needed]
        else:
            if len(neighbors) == 0:
                neighbors = [i] * needed
            else:
                while len(neighbors) < needed:
                    neighbors.extend(neighbors)
                neighbors = neighbors[:needed]
        
        # Node mục tiêu luôn ở cuối cùng
        sub_indices = neighbors + [i]
        
        # Đưa feature vào mảng
        final_feats[i] = features[sub_indices]

    # 4. Lưu file .npy vào data_structured/
    feat_file = str(DATA_STRUCTURED / f"{output_name}_feature_{subgraph_size+1}_1.npy")
    label_file = str(DATA_STRUCTURED / f"{output_name}_label_{subgraph_size+1}_1.npy")
    
    np.save(feat_file, final_feats)
    np.save(label_file, labels)
    
    print(f"Done! Saved {feat_file} and {label_file}")


import numpy as np
from sklearn.metrics import average_precision_score

def metrics_at_k(y_true, y_score, k_ratio):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n = len(y_true)
    k = int(np.ceil(k_ratio * n))

    # Top-K indices (score cao nhất)
    idx = np.argsort(y_score)[::-1][:k]

    y_top = y_true[idx]
    score_top = y_score[idx]

    # Precision@K
    precision_k = y_top.sum() / k

    # Recall@K
    recall_k = y_top.sum() / y_true.sum()

    return precision_k, recall_k
