### Sử dụng Truncated SVD thay vì Full SVD để tiết kiệm bộ nhớ và thời gian tính toán 
### Dùng Full SVD tính toán chính xác nhưng sau đó cũng giảm chiều dữ liệu về d' nên kết quả gần như tương tự Truncated SVD
### Do không có dataset gốc nên code giả sử với dataset t_finance.mat

from scipy.io import loadmat
import scipy.io as sio
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

# svd_trans.py nằm trong ReproduceCode/ -> lên 1 cấp vào AnomalyGFM/ -> vào datasets
DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets for AnomalyGFM"

def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim # Đảm bảo số chiều ban đầu >= chiều mong muốn
    svd = TruncatedSVD(n_components=out_dim, n_iter=7, random_state=42)
    newdata = svd.fit_transform(data)
    return newdata

dataset_str = "t_finance"
emb_dimension = 10 # Đây chính là giá trị common dimensionality d' như trong paper
data = loadmat(str(DATASET_DIR / '{}.mat'.format(dataset_str)))

label = data['Label'] if ('Label' in data) else data['gnd']          
attr = data['Attributes'] if ('Attributes' in data) else data['X']
adj = data['Network'] if ('Network' in data) else data['A'] 

newFeat = x_svd(attr, emb_dimension)

print('Original feature shape: ', attr.shape)
print('Reduced feature shape (Unified): ', newFeat.shape)

sio.savemat(str(DATASET_DIR / '{}_svd_{}.mat'.format(dataset_str, emb_dimension)), \
            {'Network': adj, 'Label': label, 'Attributes': newFeat})