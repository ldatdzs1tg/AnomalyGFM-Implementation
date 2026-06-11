# data_raw — Raw Transaction Data

This folder contains the raw Ethereum transaction data for the BERT4ETH extension.

> 📖 **For full documentation**, see [`b4e/README.md`](../README.md).

---

## ⚠️ Extraction Required

The 4 raw CSV files (~6 GB total) are packaged in `b4e_dataset.rar`.  
**Extract before running `b4e/src/preprocess.ipynb`.**

### Using 7-Zip

```bash
7z x b4e_dataset.rar -o./
```

### Using WinRAR

Right-click `b4e_dataset.rar` → *Extract Here*

---

After extraction this folder should contain:

```
data_raw/
├── b4e_dataset.rar
├── phisher_account.txt                              ← ground truth labels
├── normal_eoa_transaction_in_slice_1000K.csv        ← ~1 GB
├── normal_eoa_transaction_out_slice_1000K.csv       ← ~5 GB
├── phisher_transaction_in.csv                       ← ~100 MB
└── phisher_transaction_out.csv                      ← ~100 MB
```
