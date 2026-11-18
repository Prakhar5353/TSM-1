import os
import glob
import math
import json
import array
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

# ==============================================================================
# 1. Configuration & Hardcoded Inputs
# ==============================================================================
TARGET_SEEDS = 19
EVENT_THRESHOLD = 100000  # minimum event count
X_MIN, X_MAX = 0.077, 2.99
Y_MIN, Y_MAX = 0.075, 2.97
KINEMATIC_LIMIT = 3.48 

WORK_DIR = f"global_output_vornoi_{TARGET_SEEDS}_1sigma"
SEED_FILE = os.path.join(WORK_DIR, f"global_cp_node_{TARGET_SEEDS}_seeds.json")
OUT_ROOT = os.path.join(WORK_DIR, "global_phase1_state.root")
OUT_PNG = os.path.join(WORK_DIR, "global_phase1_final_map.png")

if not os.path.exists(SEED_FILE):
    raise FileNotFoundError(f"Could not find {SEED_FILE}. Run global_seeder.py first!")

TREE_NAME = "DalitzEventList"
FILE_D0   = "Dtopipipi0.root"

DATA_DIR = "1sigma"


TOY_DBAR_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "Dbartopipipi0_*.root"))) # Taking input of all Toy Dbar model files 

if not TOY_DBAR_FILES:
    raise FileNotFoundError(f"No ROOT files matching 'Dbartopipipi0_*.root' found in '{DATA_DIR}/'!")

# Extract model keys dynamically (e.g. '1sigma/Dbartopipipi0_rhom_mag_dec.root' -> 'rhom_mag_dec')
model_names = [
    os.path.basename(f).replace("Dbartopipipi0_", "").replace(".root", "") 
    for f in TOY_DBAR_FILES
]

print(f"Detected {len(model_names)} toy models in '{DATA_DIR}/': {model_names}")

# ==============================================================================
# 2. Creating 50 x 50 uniform bins with bin edges aligned with resonant masses squared
# ==============================================================================
m2_rho770  = 0.77477**2
m2_f21270  = 1.275**2
m2_rho1450 = 1.44502**2

anchors = [0.0, m2_rho770, m2_f21270, m2_rho1450, 3.1]
n_total_bins = 50
lengths = np.diff(anchors)
total_length = 3.1
bins_per_segment = np.round(lengths / total_length * n_total_bins).astype(int)

while bins_per_segment.sum() < n_total_bins:
    bins_per_segment[np.argmax(lengths)] += 1
while bins_per_segment.sum() > n_total_bins:
    bins_per_segment[np.argmax(lengths)] -= 1

edges = [0.0]
for i in range(len(anchors) - 1):
    seg_edges = np.linspace(anchors[i], anchors[i+1], bins_per_segment[i] + 1)
    edges.extend(seg_edges[1:])

bin_edges = array.array('d', edges)
x_centers = [(edges[i] + edges[i+1]) / 2.0 for i in range(n_total_bins)]
y_centers = [(edges[i] + edges[i+1]) / 2.0 for i in range(n_total_bins)]

# ==============================================================================
# 3. Load Data & Generate Universal Grids
# ==============================================================================
def get_dalitz_hist(file_path, hist_name):
    df = ROOT.RDataFrame(TREE_NAME, file_path)
    cols = [str(c) for c in df.GetColumnNames()]
    
    if "_1_pi~_E" in cols:
        df = df.Alias("pip_E",  "_1_pi~_E").Alias("pip_PX", "_1_pi~_Px").Alias("pip_PY", "_1_pi~_Py").Alias("pip_PZ", "_1_pi~_Pz")
        df = df.Alias("pim_E",  "_2_pi#_E").Alias("pim_PX", "_2_pi#_Px").Alias("pim_PY", "_2_pi#_Py").Alias("pim_PZ", "_2_pi#_Pz")
        df = df.Alias("pi0_E",  "_3_pi0_E").Alias("pi0_PX", "_3_pi0_Px").Alias("pi0_PY", "_3_pi0_Py").Alias("pi0_PZ", "_3_pi0_Pz")
    elif "pim_Px" in cols:
        df = df.Alias("pip_PX", "pip_Px").Alias("pip_PY", "pip_Py").Alias("pip_PZ", "pip_Pz")
        df = df.Alias("pim_PX", "pim_Px").Alias("pim_PY", "pim_Py").Alias("pim_PZ", "pim_Pz")
        df = df.Alias("pi0_PX", "pi0_Px").Alias("pi0_PY", "pi0_Py").Alias("pi0_PZ", "pi0_Pz")

    df = df.Define("m2_pippim", "(pow(pip_E+pim_E,2) - pow(pip_PX+pim_PX,2) - pow(pip_PY+pim_PY,2) - pow(pip_PZ+pim_PZ,2))")
    df = df.Define("m2_pippi0", "(pow(pip_E+pi0_E,2) - pow(pip_PX+pi0_PX,2) - pow(pip_PY+pi0_PY,2) - pow(pip_PZ+pi0_PZ,2))")
    if "weight" not in cols: 
        df = df.Define("weight", "1.0")

    model = (hist_name, "", n_total_bins, bin_edges, n_total_bins, bin_edges)
    return df.Histo2D(model, "m2_pippim", "m2_pippi0", "weight")

print("Loading Nominal D0 and 8 independent Dbar Toy Models...")
h_plus_ptr = get_dalitz_hist(FILE_D0, "h_plus_nominal")
h_plus = h_plus_ptr.GetValue()

grid_plus = np.zeros((n_total_bins, n_total_bins), dtype=float)
for i in range(n_total_bins):
    for j in range(n_total_bins):
        grid_plus[i, j] = h_plus.GetBinContent(i + 1, j + 1)

grid_minus_dict = {}
grid_acp_dict = {}
h_minus_hists = {}

for toy_file in TOY_DBAR_FILES:
    model_name = os.path.basename(toy_file).replace("Dbartopipipi0_", "").replace(".root", "")
    h_minus_ptr = get_dalitz_hist(toy_file, f"h_minus_{model_name}")
    h_m = h_minus_ptr.GetValue()
    h_minus_hists[model_name] = h_m
    
    grid_m = np.zeros((n_total_bins, n_total_bins), dtype=float)
    grid_a = np.zeros((n_total_bins, n_total_bins), dtype=float)
    
    for i in range(n_total_bins):
        for j in range(n_total_bins):
            nm = h_m.GetBinContent(i + 1, j + 1)
            np_val = grid_plus[i, j]
            grid_m[i, j] = nm
            tot = np_val + nm
            grid_a[i, j] = (np_val - nm) / tot if tot > 0 else 0.0
            
    grid_minus_dict[model_name] = grid_m
    grid_acp_dict[model_name] = grid_a

# ==============================================================================
# 4. Universal Mask & Target Setup
# ==============================================================================
print(f"Loading {TARGET_SEEDS} master seeds and building Universal Mask...")
with open(SEED_FILE, "r") as f:
    seeds = json.load(f)

seed_info = {s["id"]: s for s in seeds}

def get_grid_indices(x, y):
    r = max(0, min(n_total_bins - 1, np.searchsorted(edges, x) - 1))
    c = max(0, min(n_total_bins - 1, np.searchsorted(edges, y) - 1))
    return r, c

assigned_map = np.full((n_total_bins, n_total_bins), -1, dtype=int)

for i in range(n_total_bins):
    for j in range(n_total_bins):
        x, y = x_centers[i], y_centers[j]
        if not (X_MIN < x < X_MAX and Y_MIN < y < Y_MAX and x + y < KINEMATIC_LIMIT): continue
        if grid_plus[i, j] <= 0: continue
            
        valid_in_all = True
        for model_name, grid_m in grid_minus_dict.items():
            if grid_m[i, j] <= 0:
                valid_in_all = False
                break
                
        if valid_in_all:
            assigned_map[i, j] = 0

current_pop = {s["id"]: 0.0 for s in seeds}
for s in seeds:
    r, c = get_grid_indices(s["x"], s["y"])
    assigned_map[r, c] = s["id"]
    current_pop[s["id"]] += grid_plus[r, c]

# ==============================================================================
# 5. Step A: 100k Threshold
# ==============================================================================
print(f"\n--- STEP A: Minimum events threshold step (Threshold: {EVENT_THRESHOLD:,} events) ---")
changed = True
iter_a = 0

while changed:
    changed = False
    iter_a += 1
    proposals = []
    
    for i in range(n_total_bins):
        for j in range(n_total_bins):
            if assigned_map[i, j] == 0:  
                best_seed = -1
                min_dist = float('inf')
                
                # Check 4-way neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_total_bins and 0 <= nj < n_total_bins:
                        neighbor_seed = assigned_map[ni, nj]
                        
                        if neighbor_seed > 0:
                            # ONLY underpopulated seeds are allowed to absorb
                            if current_pop[neighbor_seed] < EVENT_THRESHOLD:
                                s_data = seed_info[neighbor_seed]
                                dist = math.hypot(x_centers[i] - s_data['x'], y_centers[j] - s_data['y'])
                                if dist < min_dist:
                                    min_dist = dist
                                    best_seed = neighbor_seed
                                    
                if best_seed != -1:
                    proposals.append((min_dist, i, j, best_seed))
                    
    proposals.sort(key=lambda x: x[0])
    
    for dist, i, j, seed in proposals:
        if assigned_map[i, j] == 0 and current_pop[seed] < EVENT_THRESHOLD:
            assigned_map[i, j] = seed
            current_pop[seed] += grid_plus[i, j]
            changed = True

min_survivor = min(current_pop.values())
print(f"Step A completed in {iter_a} iterations. Minimum bin population is now: {min_survivor:,.0f}")

# ==============================================================================
# 6. Step B: Physics Growth (CP Parity respecting)
# ==============================================================================
print(f"\n--- STEP B: Physics Growth (CP Parity Respecting) ---")
changed = True
iter_b = 0

while changed:
    changed = False
    iter_b += 1
    proposals = []
    
    for i in range(n_total_bins):
        for j in range(n_total_bins):
            if assigned_map[i, j] == 0:
                best_seed = -1
                min_dist = float('inf')
                
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_total_bins and 0 <= nj < n_total_bins:
                        neighbor_seed = assigned_map[ni, nj]
                        
                        if neighbor_seed > 0:
                            s_data = seed_info[neighbor_seed]
                            origin_toy = s_data["origin_toy"]
                            seed_pol = s_data["acp_sign"]
                            pixel_acp = grid_acp_dict[origin_toy][i, j]
                            
                            # PHYSICS VETO ACTIVATED
                            if (seed_pol > 0 and pixel_acp < 0) or (seed_pol < 0 and pixel_acp > 0):
                                continue 
                            
                            dist = math.hypot(x_centers[i] - s_data['x'], y_centers[j] - s_data['y'])
                            if dist < min_dist:
                                min_dist = dist
                                best_seed = neighbor_seed
                                
                if best_seed != -1:
                    proposals.append((min_dist, i, j, best_seed))
                    
    proposals.sort(key=lambda x: x[0])
    
    for dist, i, j, seed in proposals:
        if assigned_map[i, j] == 0:
            assigned_map[i, j] = seed
            current_pop[seed] += grid_plus[i, j]
            changed = True

print(f"Step B stalled against CP Parity walls after {iter_b} iterations.")

# ==============================================================================
# 7. Step C: Geometric Edge Cleanup
# ==============================================================================
print(f"\n--- STEP C: Nearest Edge Geometric Cleanup ---")
changed = True
iter_c = 0

while changed:
    changed = False
    iter_c += 1
    new_assignments = {}
    
    for i in range(n_total_bins):
        for j in range(n_total_bins):
            if assigned_map[i, j] == 0:
                neighbor_seeds = set()
                
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_total_bins and 0 <= nj < n_total_bins:
                        n_seed = assigned_map[ni, nj]
                        if n_seed > 0:
                            neighbor_seeds.add(n_seed)
                            
                if neighbor_seeds:
                    # Resolve multiple edges by closest master seed origin
                    best_seed = -1
                    min_seed_dist = float('inf')
                    for n_seed in neighbor_seeds:
                        s_data = seed_info[n_seed]
                        dist = math.hypot(x_centers[i] - s_data['x'], y_centers[j] - s_data['y'])
                        if dist < min_seed_dist:
                            min_seed_dist = dist
                            best_seed = n_seed
                    if best_seed != -1:
                        new_assignments[(i, j)] = best_seed
                        
    for (i, j), seed in new_assignments.items():
        assigned_map[i, j] = seed
        current_pop[seed] += grid_plus[i, j]
        changed = True

print(f"Step C contiguous map achieved after {iter_c} iterations of edge-dilation.")

# ==============================================================================
# 8. Final Population Output
# ==============================================================================
print("\n==================================================")
print("Final Phase 1 Superbin Populations (Nominal D0):")
print("==================================================")
for b_id in sorted(current_pop.keys()):
    pop = current_pop[b_id]
    print(f"Superbin {b_id:2d}: {pop:9,.0f} events")
print("==================================================\n")

# ==============================================================================
# 9. Export Phase 1 State
# ==============================================================================
h_voronoi = h_plus.Clone("h_voronoi")
h_voronoi.SetTitle("Global Phase 1: 3-Step Contiguous Map;m^{2}(#pi^{+}#pi^{-});m^{2}(#pi^{+}#pi^{0})")
h_voronoi.Reset()

for i in range(n_total_bins):
    for j in range(n_total_bins):
        if assigned_map[i, j] > 0:
            h_voronoi.SetBinContent(i + 1, j + 1, assigned_map[i, j])

f_out = ROOT.TFile(OUT_ROOT, "RECREATE")
h_voronoi.Write("h_voronoi")  
h_plus.Write("h_Nplus_nominal")
for model_name, h_m in h_minus_hists.items():
    h_m.Write(f"h_Nminus_{model_name}")
f_out.Close()
print(f"Successfully packed Phase 1 Voronoi Map and 9 event grids into {OUT_ROOT}")

# ==============================================================================
# 10. Visualization 
# ==============================================================================
base_colors = [ROOT.kWhite, ROOT.kBlack] 
palette = [
    ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kCyan, 
    ROOT.kOrange+7, ROOT.kSpring+4, ROOT.kViolet, ROOT.kAzure+1, 
    ROOT.kTeal+3, ROOT.kPink+2, ROOT.kYellow+2, ROOT.kRed-4, 
    ROOT.kBlue-4, ROOT.kGreen-3, ROOT.kMagenta-3, ROOT.kCyan-3, 
    ROOT.kOrange-3, ROOT.kSpring-5, ROOT.kViolet-6,
    ROOT.kRed-7, ROOT.kBlue-7, ROOT.kGreen-7, ROOT.kMagenta-7,
    ROOT.kCyan-7, ROOT.kOrange-7, ROOT.kSpring-7, ROOT.kViolet-7,
    ROOT.kAzure+6, ROOT.kTeal-5, ROOT.kPink-5, ROOT.kYellow-5
]
final_colors = base_colors + palette
custom_colors = array.array('i', final_colors[:TARGET_SEEDS + 2])
ROOT.gStyle.SetPalette(len(custom_colors), custom_colors)

g_seeds = ROOT.TGraph()
g_seeds.SetMarkerStyle(29)
g_seeds.SetMarkerSize(2.5)
g_seeds.SetMarkerColor(ROOT.kWhite)
for idx, s in enumerate(seeds):
    g_seeds.SetPoint(idx, s['x'], s['y'])

c1 = ROOT.TCanvas("c_map", "", 800, 800)
c1.SetLeftMargin(0.15)
c1.SetRightMargin(0.18)

h_voronoi.SetMinimum(-1) 
h_voronoi.SetMaximum(TARGET_SEEDS)
h_voronoi.SetContour(TARGET_SEEDS + 2)
h_voronoi.Draw("COLZ")
g_seeds.Draw("P SAME")

c1.SaveAs(OUT_PNG)
print(f"Saved Global Phase 1 contiguous visualization to {OUT_PNG}")