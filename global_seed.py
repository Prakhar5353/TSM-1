import os
import glob
import math
import json
import array
import subprocess
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)

# ==============================================================================
# 1. Configuration & Hardcoded Inputs
# ==============================================================================
TARGET_SEEDS = 19 # Equal to final bins required

# Dynamic D_MIN hyperparameters
D_MIN_MAX = 0.63   # Exclusion radius for the sparsest valid regions
D_MIN_MIN = 0.12  # Exclusion radius for the densest resonance peaks

# Scoring Hyperparameters
GAMMA = 0.7   # Density dampener (0.5 = square root)
BETA = 0.7   # Weight applied to CP Asymmetry relative to density

EVENT_THRESHOLD = 1000
MARGIN = 0.1
X_MIN, X_MAX = 0.077, 2.99
Y_MIN, Y_MAX = 0.075, 2.97
KINEMATIC_LIMIT = 3.48 



OUT_DIR = f"global_output_vornoi_{TARGET_SEEDS}_1sigma"
os.makedirs(OUT_DIR, exist_ok=True)

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

# Resonance lines for visual reference
m2_rho770  = 0.77477**2
m2_f21270  = 1.275**2
m2_rho1450 = 1.44502**2

resonances = [
    {"name": "#rho(770)",    "m2": m2_rho770,  "color": ROOT.kRed,   "has_horizontal": True},
    {"name": "f_{2}(1270)",  "m2": m2_f21270,  "color": ROOT.kWhite, "has_horizontal": False},
    {"name": "#rho(1450)",   "m2": m2_rho1450, "color": ROOT.kCyan,  "has_horizontal": True},
]

# ==============================================================================
# 2. Creating 50 x 50 uniform bins with bin edges aligned with resonant masses squared
# ==============================================================================
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

# ==============================================================================
# 3. RDataFrame Helper
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

EFF_FILE = "efficiency/dalitz_efficiency_map_ampgen_denom.root" # Taking input of efficiency file here

eff_file = ROOT.TFile.Open(EFF_FILE, "READ")

if not eff_file or eff_file.IsZombie():
    raise RuntimeError(f"Could not open efficiency file: {EFF_FILE}")

h_eff = eff_file.Get("h_efficiency_map")

if not h_eff:
    raise RuntimeError("Could not find h_eff_map in efficiency ROOT file")

if (h_eff.GetNbinsX() != n_total_bins or
        h_eff.GetNbinsY() != n_total_bins):
    raise RuntimeError(
        "Efficiency histogram dimensions do not match the seed-selector grid"
    )

# Verify that both efficiency-map axes use exactly the same variable-width
# bin edges as the Dalitz histograms produced by this script.
expected_edges = np.asarray(edges, dtype=float)

for axis_name, axis in (("X", h_eff.GetXaxis()), ("Y", h_eff.GetYaxis())):
    actual_edges = np.array(
        [axis.GetBinLowEdge(k) for k in range(1, n_total_bins + 2)],
        dtype=float
    )
    if not np.allclose(actual_edges, expected_edges, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            f"Efficiency histogram {axis_name}-axis edges do not match "
            "the seed-selector Dalitz grid"
        )

# Detach the histogram from its ROOT file so it remains valid after closing it.
h_eff.SetDirectory(0)
eff_file.Close()

# Do not silently clip efficiencies. Values above one are retained for
# validation, while negative or non-finite values are rejected.
eff_values = np.array(
    [[h_eff.GetBinContent(i, j)
      for j in range(1, n_total_bins + 1)]
     for i in range(1, n_total_bins + 1)],
    dtype=float
)

if not np.all(np.isfinite(eff_values)):
    raise RuntimeError("Efficiency map contains non-finite bin contents")

if np.any(eff_values < 0.0):
    raise RuntimeError("Efficiency map contains negative bin contents")

n_eff_above_one = int(np.count_nonzero(eff_values > 1.0))
if n_eff_above_one:
    print(
        f"[WARNING] Efficiency map contains {n_eff_above_one} bins "
        "with efficiency > 1. They are retained without clipping."
    )

print(f"Loaded and validated reconstruction efficiency map: {EFF_FILE}")

# ==============================================================================
# 4. Creating global pool of all possible seed positions
# ==============================================================================
global_pool = []
saved_hacp_grids = {}

h_plus_ptr = get_dalitz_hist(FILE_D0, "h_plus_nominal")
h_plus = h_plus_ptr.GetValue()

print("Generating Mega-Pool from all independent toy models...")

for toy_file in TOY_DBAR_FILES:
    if not os.path.exists(toy_file):
        print(f"[WARNING] Skipping missing file: {toy_file}")
        continue
        
    model_name = os.path.basename(toy_file).replace("Dbartopipipi0_", "").replace(".root", "")
    
    h_minus_ptr = get_dalitz_hist(toy_file, f"h_minus_{model_name}")
    h_minus = h_minus_ptr.GetValue()
    
    h_acp = h_plus.Clone(f"h_acp_{model_name}")
    h_acp.Reset()
    h_acp.SetDirectory(0) 
    
    for i in range(1, n_total_bins + 1):
        for j in range(1, n_total_bins + 1):
            n_p = h_plus.GetBinContent(i, j)
            n_m = h_minus.GetBinContent(i, j)
            total = n_p + n_m
            
            x_c = h_plus.GetXaxis().GetBinCenter(i)
            y_c = h_plus.GetYaxis().GetBinCenter(j)
            
            if total > 0:
                acp = (n_p - n_m) / total
                h_acp.SetBinContent(i, j, acp)
            else:
                h_acp.SetBinContent(i, j, 0.0)
                
            # Primary filter for the center candidate
            if total >= EVENT_THRESHOLD and (X_MIN + MARGIN < x_c < X_MAX - MARGIN) and (Y_MIN + MARGIN < y_c < Y_MAX - MARGIN) and (x_c + y_c < KINEMATIC_LIMIT - MARGIN):
                
                # 3x3 Neighborhood Evaluation
                valid_count = 0
                sum_events = 0.0
                
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni = i + di
                        nj = j + dj
                        
                        # Only check if neighbor is physically within the grid array
                        if 1 <= ni <= n_total_bins and 1 <= nj <= n_total_bins:
                            n_p_n = h_plus.GetBinContent(ni, nj)
                            n_m_n = h_minus.GetBinContent(ni, nj)
                            tot_n = n_p_n + n_m_n
                            
                            xn = h_plus.GetXaxis().GetBinCenter(ni)
                            yn = h_plus.GetYaxis().GetBinCenter(nj)
                            
                            is_valid = (tot_n >= EVENT_THRESHOLD) and \
                                       (X_MIN + MARGIN < xn < X_MAX - MARGIN) and \
                                       (Y_MIN + MARGIN < yn < Y_MAX - MARGIN) and \
                                       (xn + yn < KINEMATIC_LIMIT - MARGIN)
                            
                            if is_valid:
                                valid_count += 1
                                sum_events += tot_n
                
                # Rejection Rule: 3 or more invalid pixels (valid_count < 7)
                invalid_count = 9 - valid_count
                if invalid_count >= 3:
                    continue
                    
                avg_events = sum_events / valid_count
                
                global_pool.append({
                    'x': float(round(x_c, 3)), 
                    'y': float(round(y_c, 3)), 
                    'acp': acp, 
                    'n_tot': total,
                    'avg_events': avg_events,
                    'bin_x': i,
                    'bin_y': j,
                    'origin_toy': model_name
                })
    
    saved_hacp_grids[model_name] = h_acp

print(f"Total candidates in Global Mega-Pool: {len(global_pool)}")

# ==============================================================================
# 5. Scoring each seed and calculation of its D_MIN
# ==============================================================================
# Pass 1: Find Global Maximums for normalization & log mappings
n_max = max([cand['n_tot'] for cand in global_pool]) if global_pool else 1.0
acp_max = max([abs(cand['acp']) for cand in global_pool]) if global_pool else 1.0

min_avg_events = min([cand['avg_events'] for cand in global_pool]) if global_pool else 1.0
max_avg_events = max([cand['avg_events'] for cand in global_pool]) if global_pool else 2.0

log_min_avg = math.log10(min_avg_events)
log_max_avg = math.log10(max_avg_events)

print(f"\n--- Metric Normalization Anchors ---")
print(f"Global Max Events in a pixel: {n_max:,.0f}")
print(f"Global Max |A_cp| in a pixel: {acp_max:.4f}")
print(f"Neighborhood Density Range: {min_avg_events:.0f} to {max_avg_events:.0f} events")
print(f"Parameters: GAMMA={GAMMA}, BETA={BETA}, D_MIN Range=[{D_MIN_MIN}, {D_MIN_MAX}]")
print("------------------------------------\n")

# Pass 2: Calculate unified Score and dynamic d_min for each candidate
for cand in global_pool:
    # 1. Scoring
    # 1. Scoring with reconstruction efficiency
    norm_density = cand['n_tot'] / n_max
    norm_acp = abs(cand['acp']) / acp_max if acp_max > 0.0 else 0.0

    # Use the original integer ROOT bin indices. This avoids assigning a
    # candidate to a neighbouring efficiency bin because its plotted
    # coordinates were rounded to three decimal places.
    efficiency = h_eff.GetBinContent(cand['bin_x'], cand['bin_y'])

    # Candidates in zero-efficiency bins cannot be selected as seeds.
    cand['efficiency'] = efficiency

    if efficiency <= 0.0:
        cand['score'] = 0.0
        cand['d_min'] = D_MIN_MAX
        continue

    # Original score multiplied by that pixel's reconstruction efficiency.
    original_score = (norm_density ** GAMMA) + BETA * norm_acp
    cand['score'] = original_score * efficiency
    
    # 2. Dynamic D_MIN (Logarithmic Interpolation)
    log_avg = math.log10(cand['avg_events'])
    if log_max_avg > log_min_avg:
        fraction = (log_avg - log_min_avg) / (log_max_avg - log_min_avg)
        # Fraction 0 -> D_MIN_MAX (0.3), Fraction 1 -> D_MIN_MIN (0.06)
        cand['d_min'] = D_MIN_MAX - fraction * (D_MIN_MAX - D_MIN_MIN)
    else:
        cand['d_min'] = D_MIN_MAX

# Remove zero-efficiency candidates entirely, so they cannot be selected
# even if fewer than TARGET_SEEDS positive-efficiency candidates exist.
global_pool = [
    cand for cand in global_pool
    if cand['efficiency'] > 0.0
]

if not global_pool:
    raise RuntimeError(
        "No seed candidates remain after applying the efficiency requirement"
    )

# Sort the entire mega-pool globally by the efficiency-weighted score.
global_pool.sort(key=lambda item: item['score'], reverse=True)

accepted_seeds = []

for cand in global_pool:
    if len(accepted_seeds) >= TARGET_SEEDS:
        break
        
    too_close = False
    for s in accepted_seeds:
        dist = math.hypot(cand['x'] - s['x'], cand['y'] - s['y'])
        # Option 1A: Strict Mutual Exclusion
        required_dist = max(cand['d_min'], s['d_min'])
        
        if dist < required_dist:
            too_close = True
            break
            
    if not too_close:
        accepted_seeds.append(cand)

if len(accepted_seeds) < TARGET_SEEDS:
    print(f"[WARNING] Could only pack {len(accepted_seeds)} seeds. Space is highly constrained.")

final_seeds = [{"id": idx+1, "x": s['x'], "y": s['y'], "acp_sign": np.sign(s['acp']), "acp_magnitude": abs(s['acp']), "efficiency": s['efficiency'], "score": s['score'], "d_min_assigned": round(s['d_min'], 3), "origin_toy": s['origin_toy']} for idx, s in enumerate(accepted_seeds)]

out_json = os.path.join(OUT_DIR, f"global_cp_node_{TARGET_SEEDS}_seeds.json")
with open(out_json, "w") as f:
    json.dump(final_seeds, f, indent=4)
print(f"Successfully locked {len(accepted_seeds)} universal consensus seeds using Dynamic Metric.")
print(f"Coordinates saved to {out_json}")

# ==============================================================================
# 6. Global Visualizations (Plot Master Seeds onto Maps)
# ==============================================================================
persistent_objects = []  

def draw_resonance_lines_and_legend():
    leg = ROOT.TLegend(0.55, 0.68, 0.82, 0.85)
    leg.SetBorderSize(1)
    leg.SetFillColor(ROOT.kBlack)
    leg.SetTextColor(ROOT.kWhite)
    leg.SetTextSize(0.035)
    
    for res in resonances:
        m2 = res["m2"]
        color = res["color"]
        
        v_line = ROOT.TLine(m2, 0.0, m2, 3.1)
        v_line.SetLineColor(color)
        v_line.SetLineWidth(2)
        v_line.SetLineStyle(2)
        v_line.Draw("SAME")
        persistent_objects.append(v_line)
        
        if res["has_horizontal"]:
            h_line = ROOT.TLine(0.0, m2, 3.1, m2)
            h_line.SetLineColor(color)
            h_line.SetLineWidth(2)
            h_line.SetLineStyle(2)
            h_line.Draw("SAME")
            persistent_objects.append(h_line)
            
        leg.AddEntry(v_line, res["name"], "l")
        
    leg.Draw("SAME")
    persistent_objects.append(leg)

# A. Validation on A_cp Toy Maps
for model_name, h_acp in saved_hacp_grids.items():
    c = ROOT.TCanvas(f"c_single_{model_name}", f"Max-Node Seed Validation {model_name}", 800, 800)
    c.SetLeftMargin(0.15)
    c.SetRightMargin(0.15)

    g_seeds = ROOT.TGraph()
    g_seeds.SetMarkerStyle(29)  
    g_seeds.SetMarkerSize(2.5)
    g_seeds.SetMarkerColor(ROOT.kPink + 1)

    for idx, s in enumerate(final_seeds):
        g_seeds.SetPoint(idx, s['x'], s['y'])

    ROOT.gStyle.SetPalette(ROOT.kTemperatureMap)
    h_acp.SetTitle(f"Dalitz A_{{cp}} Map ({model_name}) with {len(accepted_seeds)} Master Seeds;m^{{2}}(#pi^{{+}}#pi^{{-}}) [GeV^{{2}}/c^{{4}}];m^{{2}}(#pi^{{+}}#pi^{{0}}) [GeV^{{2}}/c^{{4}}]")

    h_acp.SetMinimum(-0.1)
    h_acp.SetMaximum(0.1)
    h_acp.Draw("COLZ")
    draw_resonance_lines_and_legend()
    g_seeds.Draw("P SAME")
    c.Update()

    out_img = os.path.join(OUT_DIR, f"global_seed_validation_{model_name}.png")
    c.SaveAs(out_img)
    print(f"Saved validation map for {model_name} to {out_img}")

# B. Validation on Nominal Event Density Map
c_ev = ROOT.TCanvas("c_events_density", "Event Density Seed Validation", 800, 800)
c_ev.SetLeftMargin(0.15)
c_ev.SetRightMargin(0.15)

# Reset seeds to Red so they show up clearly on the event map
g_seeds_ev = ROOT.TGraph()
g_seeds_ev.SetMarkerStyle(29)  
g_seeds_ev.SetMarkerSize(2.5)
g_seeds_ev.SetMarkerColor(ROOT.kRed)
for idx, s in enumerate(final_seeds):
    g_seeds_ev.SetPoint(idx, s['x'], s['y'])

ROOT.gStyle.SetPalette(ROOT.kBird) # Distinct palette from Temperature Map
h_plus.SetTitle(f"D^{{0}} Nominal Event Density with {len(accepted_seeds)} Seeds;m^{{2}}(#pi^{{+}}#pi^{{-}}) [GeV^{{2}}/c^{{4}}];m^{{2}}(#pi^{{+}}#pi^{{0}}) [GeV^{{2}}/c^{{4}}]")
h_plus.Draw("COLZ")
draw_resonance_lines_and_legend()
g_seeds_ev.Draw("P SAME")

out_ev = os.path.join(OUT_DIR, "global_seed_validation_density_nominal.png")
c_ev.SaveAs(out_ev)
print(f"Saved validation map for Nominal Event Density to {out_ev}")

# ==============================================================================
# 7. Pipeline Execution: Trigger Global Phase 1 & 2 (Running just this script ensures that entire script will run)
# ==============================================================================
print("\n==================================================")
print("Initiating Global Automated Clustering Pipeline...")
print("==================================================")

try:
    print("\n>>> Running Global Phase 1: Multi-Toy Guided Adjacency Growth...")
    subprocess.run(["python", "global_phase1.py"], check=True)
    print(">>> Global Phase 1 completed successfully.")

    print("\n>>> Running Global Phase 2: Consensus Cellular Automaton...")
    subprocess.run(["python", "global_phase2.py"], check=True)
    print(">>> Global Phase 2 completed successfully.")

    print("\n==================================================")
    print("Full Global Pipeline Execution Complete!")
    print(f"Check your output directory: {OUT_DIR} for the final maps.")
    print("==================================================")

except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Pipeline aborted. A script failed with exit code {e.returncode}.")
    print(f"Failed command: {' '.join(e.cmd)}")
except FileNotFoundError:
    print("\n[ERROR] Could not find 'global_phase1.py', 'global_phase2.py', or 'global_phase3.py'. Make sure they are in the same directory.")