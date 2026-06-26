import numpy as np
import pandas as pd

filepaths = ["new_testX1.dat","new_testY1.dat","new_testZ1.dat"]

dat = []

for file in filepaths:
    raw = pd.read_csv(file, sep="\t",header=None, usecols=[1, 2, 3], names=["X", "Y", "Z"])
    raw['B'] = np.linalg.norm(raw.values, axis=1)
    raw['DeltaB'] = raw['B'] - raw['B'].iloc[0]
    dat.append(raw)
    print(min(np.abs(raw["DeltaB"])))

start_idx = 0
end_idx = len(dat[0])

for d in dat:
    # Find where absolute DeltaB is >= 100
    valid_indices = np.where(np.abs(d['DeltaB']) >= 100)[0]
    
    if len(valid_indices) > 0:
        # We want the 'tightest' window that works for all files
        start_idx = max(start_idx, valid_indices[0])
        end_idx = min(end_idx, valid_indices[-1])


interleaved_list = []
# Slice the values using [start_idx : end_idx + 1]
for row_x, row_y, row_z in zip(dat[0].values[start_idx:end_idx+1], 
                               dat[1].values[start_idx:end_idx+1], 
                               dat[2].values[start_idx:end_idx+1]):
    interleaved_list.append(row_x[:3])
    interleaved_list.append(row_y[:3])
    interleaved_list.append(row_z[:3])

diff_list = []
# Do the same for the DeltaB values
for row_values in zip(*(d['DeltaB'].values[start_idx:end_idx+1] for d in dat)):
    for i, val in enumerate(row_values):
        arr = np.zeros(3)
        arr[i] = val
        diff_list.append(arr)


# dimf = np.array([
#     [29000,-5000,42500],
#     [300,-49000,11000],
#     [35411,2500,-35000]])

# applied= np.array([[1,0,0],
#                   [0,1,0],
#                   [0,0,1]])

#Dim_field un Applied_field jāievada kā sarakstskas sastāv no np array (x,y,z)
#mērījumu skaitam jādalās ar 3
#prefferably katri 3 savā starpā ortogonāli or close to that
def Rotacijas_matrx(Dim_field, Applied_field):
# 1. Reshape data into N x 3 matrices
    # Instead of triplets, we treat all data points as a single cloud
    A = np.array(Dim_field)      # Dimensions: (N, 3)
    B = np.array(Applied_field)  # Dimensions: (N, 3)

    # 2. Solve the Orthogonal Procrustes Problem globally
    # H = A^T * B
    H = A.T @ B
    
    # Standard SVD approach
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt  # Note: The order depends on your vector orientation
    
    # 3. Ensure a right-handed coordinate system (no reflections)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # 4. Error Analysis
    # Apply the rotation to all points at once
    predictions = A @ R.T # Or R @ A.T depending on your data shape
    errors = np.linalg.norm(predictions - B, axis=1)
    
    # print(f"Global Mean Error: {np.mean(errors):.6f}")
    # print(f"Global Std Dev: {np.std(errors):.6f}")
    
    return R


print(Rotacijas_matrx(interleaved_list,diff_list))