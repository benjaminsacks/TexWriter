import numpy as np
import os

data_dir = 'data/processed_latex'
x_len_path = os.path.join(data_dir, 'x_len.npy')

if not os.path.exists(x_len_path):
    print(f"File not found: {x_len_path}")
    exit(1)

x_len = np.load(x_len_path)

print(f"Total samples: {len(x_len)}")
print(f"Min length: {np.min(x_len)}")
print(f"Max length: {np.max(x_len)}")
print(f"Mean length: {np.mean(x_len)}")
print(f"Median length: {np.median(x_len)}")
print("-" * 20)
print("Percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"{p}th percentile: {np.percentile(x_len, p)}")
