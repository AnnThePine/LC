import numpy as np
import matplotlib.pyplot as plt
from main_func import cetri_centri
import matplotlib.cm as cm

# Constants from your original code
lauks = 46
ik_pec_lenkis = 1
tol = 5
alpha_min = -90
alpha_max = 90
beta_min = -90
beta_max = 90

def merge_close_peaks(energies):
    """Merge peaks that are within `tol` of each other."""
    if len(energies) == 0:
        return []
    energies = np.sort(energies)
    merged = [energies[0]]
    for e in energies[1:]:
        if abs(e - merged[-1]) <= tol:
            merged[-1] = (merged[-1] + e) / 2
        else:
            merged.append(e)
    return merged

def canonical_key(e, tol=1e-6):
    """Canonical key for eigenvalue array (same or reversed)."""
    e_rounded = np.round(np.array(e) / tol) * tol
    rev_rounded = e_rounded[::-1]
    if tuple(e_rounded) < tuple(rev_rounded):
        return tuple(e_rounded)
    else:
        return tuple(rev_rounded)

# 1. Collect all points
all_points = []
for alpha in range(alpha_min, alpha_max + 1, ik_pec_lenkis):
    n_beta = max(1, int((beta_max - beta_min) / ik_pec_lenkis * np.cos(np.deg2rad(alpha))))
    for beta in np.linspace(beta_min, beta_max, n_beta, endpoint=False):
        model_energies = cetri_centri(np.array([alpha, beta, lauks]), tikaienergijas=True)
        merged_peaks = merge_close_peaks(model_energies)
        all_points.append({
            "alpha_deg": alpha,
            "beta_deg": beta,
            "energies": np.array(merged_peaks)
        })

# 2. Group all points by key
groups = {}
for p in all_points:
    key = canonical_key(p["energies"], tol=1e-6)
    groups.setdefault(key, []).append(p)

# 3. Filter groups that will be plotted
plotted_groups = []
for key, pts in groups.items():
    pts_in_interval = [p for p in pts if 0 <= p["alpha_deg"] <= 90]
    if len(pts_in_interval) >= 2:
        plotted_groups.append(pts_in_interval)

# 4. Plotting
plt.figure(figsize=(10, 8))

# Background all points in gray
plt.scatter([p["alpha_deg"] for p in all_points],
            [p["beta_deg"] for p in all_points],
            color="lightgray", s=5)

# Plot the filtered groups with a colormap
cmap = cm.get_cmap('hsv', len(plotted_groups))
for idx, pts in enumerate(plotted_groups):
    color = cmap(idx)
    plt.scatter([p["alpha_deg"] for p in pts],
                [p["beta_deg"] for p in pts],
                color=color, s=25,
                label=f"Group {idx+1} ({len(pts)} pts)")

plt.xlabel("Alpha (deg)")
plt.ylabel("Beta (deg)")
plt.title("All eigenvalue-matching groups in α∈[0,90]")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.show()