import os
import glob
import math
import json
import array
import numpy as np
from scipy.ndimage import label
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# ==============================================================================
# 1. Configuration & Hardcoded Inputs
# ==============================================================================
TARGET_SEEDS = 19
MIN_EVENTS_PER_KIND = 100000 

# Population term 
ALPHA_DOWNHILL = 120.0         # Aggressive pressure to drain big bins
ALPHA_UPHILL = 240.0         # Heavy penalty to prevent uphill flow

TARGET_EVENTS = 30e6/TARGET_SEEDS

WORK_DIR = f"global_output_vornoi_{TARGET_SEEDS}_1sigma"
STATE_FILE = os.path.join(WORK_DIR, "global_phase1_state.root")
SEED_FILE = os.path.join(WORK_DIR, f"global_cp_node_{TARGET_SEEDS}_seeds.json")

if not os.path.exists(STATE_FILE):
    raise FileNotFoundError(f"Could not find {STATE_FILE}. Run global_phase1.py first!")

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


model_names = [os.path.basename(f).replace("Dbartopipipi0_", "").replace(".root", "") for f in TOY_DBAR_FILES]

# ==============================================================================
# 2. Load Universal Phase 1 State & Event Grids
# ==============================================================================
f_in = ROOT.TFile(STATE_FILE, "READ")
h_voronoi_in = f_in.Get("h_voronoi")
h_Nplus_in   = f_in.Get("h_Nplus_nominal")

h_voronoi_in.SetDirectory(0)
h_Nplus_in.SetDirectory(0)

GRID_N = h_voronoi_in.GetNbinsX()
grid_map   = np.zeros((GRID_N, GRID_N), dtype=int)
grid_nplus = np.zeros((GRID_N, GRID_N), dtype=float)

grid_nminus_dict = {}
for m_name in model_names:
    h_nm = f_in.Get(f"h_Nminus_{m_name}")
    h_nm.SetDirectory(0)
    
    nm_arr = np.zeros((GRID_N, GRID_N), dtype=float)
    for i in range(1, GRID_N + 1):
        for j in range(1, GRID_N + 1):
            nm_arr[i-1, j-1] = h_nm.GetBinContent(i, j)
    grid_nminus_dict[m_name] = nm_arr

f_in.Close()

for i in range(1, GRID_N + 1):
    for j in range(1, GRID_N + 1):
        grid_map[i-1, j-1]   = int(h_voronoi_in.GetBinContent(i, j))
        grid_nplus[i-1, j-1] = h_Nplus_in.GetBinContent(i, j)

active_bins = sorted(list(set(grid_map.flatten()) - {0, -1}))
print(f"Loaded Phase 1 state with {len(active_bins)} active superbins.")

# ==============================================================================
# 2.1 Load and validate the pixel-by-pixel reconstruction efficiency map
# ==============================================================================
EFF_FILE = "efficiency/dalitz_efficiency_map_ampgen_denom.root"

f_eff = ROOT.TFile.Open(EFF_FILE, "READ")
if not f_eff or f_eff.IsZombie():
    raise RuntimeError(f"Could not open efficiency file: {EFF_FILE}")

h_eff = f_eff.Get("h_efficiency_map")
if not h_eff:
    f_eff.Close()
    raise RuntimeError(f"Could not find h_eff_map in {EFF_FILE}")

if h_eff.GetNbinsX() != GRID_N or h_eff.GetNbinsY() != GRID_N:
    f_eff.Close()
    raise RuntimeError(
        "Efficiency-map dimensions do not match the Phase 1 grid: "
        f"efficiency=({h_eff.GetNbinsX()}, {h_eff.GetNbinsY()}), "
        f"Phase 1=({GRID_N}, {GRID_N})"
    )

for axis_name, eff_axis, state_axis in (
    ("X", h_eff.GetXaxis(), h_voronoi_in.GetXaxis()),
    ("Y", h_eff.GetYaxis(), h_voronoi_in.GetYaxis())
):
    for edge_idx in range(1, GRID_N + 2):
        eff_edge = eff_axis.GetBinLowEdge(edge_idx)
        state_edge = state_axis.GetBinLowEdge(edge_idx)
        if not math.isclose(eff_edge, state_edge, rel_tol=0.0, abs_tol=1e-8):
            f_eff.Close()
            raise RuntimeError(
                f"Efficiency-map {axis_name}-axis edges do not match "
                f"Phase 1 at edge {edge_idx}: {eff_edge} != {state_edge}"
            )

grid_eff = np.zeros((GRID_N, GRID_N), dtype=float)
for i in range(1, GRID_N + 1):
    for j in range(1, GRID_N + 1):
        grid_eff[i-1, j-1] = h_eff.GetBinContent(i, j)

h_eff.SetDirectory(0)
f_eff.Close()

if not np.all(np.isfinite(grid_eff)):
    raise RuntimeError("Efficiency map contains non-finite bin contents")
if np.any(grid_eff < 0.0):
    raise RuntimeError("Efficiency map contains negative bin contents")

n_eff_zero = int(np.count_nonzero(grid_eff == 0.0))
n_eff_above_one = int(np.count_nonzero(grid_eff > 1.0))
print(f"Loaded pixel-by-pixel efficiency map: {EFF_FILE}")
print(f"Efficiency bins: zero={n_eff_zero}, above one={n_eff_above_one}")
if n_eff_above_one:
    print("[WARNING] Some efficiency bins exceed 1; retaining their values without clipping.")

# Load seeds for visualization overlays
seeds = []
if os.path.exists(SEED_FILE):
    with open(SEED_FILE, "r") as f:
        seeds = json.load(f)

# ==============================================================================
# 3. Helper Functions: Significance & Topology
# ==============================================================================
def calc_bin_significance(mask, nplus, nminus):
    """
    Efficiency-weighted significance for one superbin.

    Apply each pixel's efficiency to its N+ and N- yields, sum the weighted
    yields within the superbin, and calculate the significance from those
    superbin totals.
    """
    if not np.any(mask):
        return 0.0

    weighted_nplus = float(np.sum(grid_eff[mask] * nplus[mask]))
    weighted_nminus = float(np.sum(grid_eff[mask] * nminus[mask]))

    total = weighted_nplus + weighted_nminus
    if total <= 0.0:
        return 0.0

    return abs(weighted_nplus - weighted_nminus) / math.sqrt(total)


def get_model_significance(mapping, nplus, nminus, bins):
    total_s = 0.0
    for b in bins:
        mask = (mapping == b)
        total_s += calc_bin_significance(mask, nplus, nminus)
    return total_s


def count_shared_edges(mapping, r, c, target_bin):
    count = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_N and 0 <= nc < GRID_N:
            if mapping[nr, nc] == target_bin:
                count += 1
    return count

def is_bin_edge_connected(mapping, bin_id):
    mask = (mapping == bin_id)
    if not np.any(mask):
        return True
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    _, num_features = label(mask, structure=struct)
    return num_features == 1

# ==============================================================================
# 4. Phase 2: Multi-Model Consensus Cellular Automaton
# ==============================================================================
print(f"Starting Global Automaton (Alpha Downhill={ALPHA_DOWNHILL}, Alpha Uphill={ALPHA_UPHILL})...")

current_model_s = {m: get_model_significance(grid_map, grid_nplus, grid_nminus_dict[m], active_bins) for m in model_names}
current_agg_s = sum(current_model_s.values())

exchange_steps = [0]
history_aggregate = [current_agg_s]
history_models = {m: [current_model_s[m]] for m in model_names}

max_iterations = 100
iteration = 0
changed = True
total_exchanges = 0

while changed and iteration < max_iterations:
    changed = False
    iteration += 1
    moves_made = 0
    
    for r in range(GRID_N):
        for c in range(GRID_N):
            bin_a = grid_map[r, c]
            if bin_a <= 0:
                continue
                
            neighbor_bins = set()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_N and 0 <= nc < GRID_N:
                    nb = grid_map[nr, nc]
                    if nb > 0 and nb != bin_a:
                        neighbor_bins.add(nb)
            
            best_target_bin = None
            best_delta_net = 0.0
            best_delta_s_total = 0.0
            best_pressure_reward = 0.0
            best_model_s_updates = {}
            
            for bin_b in neighbor_bins:
                # Topology: Prevent tentacle like structures
                if count_shared_edges(grid_map, r, c, bin_b) < 2:
                    continue
                    
                mask_a = (grid_map == bin_a)
                mask_b = (grid_map == bin_b)
                
                # plus_a and plus_b strictly map to D0 Nominal events (grid_nplus)
                plus_a = np.sum(grid_nplus[mask_a])
                plus_b = np.sum(grid_nplus[mask_b])
                cell_plus = grid_nplus[r, c]
                
                plus_a_new = plus_a - cell_plus
                plus_b_new = plus_b + cell_plus
                
                # Global N+ Event Threshold (Vetoes swap if it breaks the 100k rule)
                if plus_a_new < MIN_EVENTS_PER_KIND or plus_b_new < MIN_EVENTS_PER_KIND:
                    continue
                    
                valid_threshold = True
                delta_s_total = 0.0
                model_s_updates = {}
                
                # Evaluate across all 8 toys
                for m_name in model_names:
                    grid_nm = grid_nminus_dict[m_name]
                    minus_a = np.sum(grid_nm[mask_a])
                    minus_b = np.sum(grid_nm[mask_b])
                    cell_minus = grid_nm[r, c]
                    
                    minus_a_new = minus_a - cell_minus
                    minus_b_new = minus_b + cell_minus
                    
                    # Universal N- Event Threshold (Strict AND)
                    if minus_a_new < MIN_EVENTS_PER_KIND or minus_b_new < MIN_EVENTS_PER_KIND:
                        valid_threshold = False
                        break
                        
                    # Evaluate both superbins using their pixel-specific efficiencies.
                    mask_a_new = mask_a.copy()
                    mask_b_new = mask_b.copy()
                    mask_a_new[r, c] = False
                    mask_b_new[r, c] = True
                    
                    s_a_old = calc_bin_significance(mask_a, grid_nplus, grid_nm)
                    s_b_old = calc_bin_significance(mask_b, grid_nplus, grid_nm)
                    s_a_new = calc_bin_significance(mask_a_new, grid_nplus, grid_nm)
                    s_b_new = calc_bin_significance(mask_b_new, grid_nplus, grid_nm)
                    
                    delta_s_model = (s_a_new + s_b_new) - (s_a_old + s_b_old)
                    delta_s_total += delta_s_model
                    model_s_updates[m_name] = current_model_s[m_name] + delta_s_model
                    
                if not valid_threshold:
                    continue
                
                # Option 1: Squared Fractional Deviation (Continuous Global Gradient)
                p_a_old = ((plus_a - TARGET_EVENTS) / TARGET_EVENTS) ** 2
                p_b_old = ((plus_b - TARGET_EVENTS) / TARGET_EVENTS) ** 2
                
                p_a_new = ((plus_a_new - TARGET_EVENTS) / TARGET_EVENTS) ** 2
                p_b_new = ((plus_b_new - TARGET_EVENTS) / TARGET_EVENTS) ** 2
                
                raw_pressure = (p_a_old + p_b_old) - (p_a_new + p_b_new)
                
                if raw_pressure > 0:
                    pressure_reward = ALPHA_DOWNHILL * raw_pressure  # Move reduces global variance
                else:
                    pressure_reward = ALPHA_UPHILL * raw_pressure    # Move increases global variance
                    
                delta_net = delta_s_total + pressure_reward
                
                # Topology: Connectivity check
                grid_map[r, c] = bin_b 
                connected = is_bin_edge_connected(grid_map, bin_a)
                grid_map[r, c] = bin_a 
                
                if not connected:
                    continue
                
                # Compare unified objective function
                if delta_net > best_delta_net:
                    best_delta_net = delta_net
                    best_delta_s_total = delta_s_total
                    best_pressure_reward = pressure_reward
                    best_target_bin = bin_b
                    best_model_s_updates = model_s_updates
            
            # Execute consensus best move
            if best_target_bin is not None:
                total_exchanges += 1
                grid_map[r, c] = best_target_bin
                
                current_agg_s += best_delta_s_total
                for m_name in model_names:
                    current_model_s[m_name] = best_model_s_updates[m_name]
                
                # Explicitly append only the significance (pure physics) to the graphs
                exchange_steps.append(total_exchanges)
                history_aggregate.append(current_agg_s)
                for m_name in model_names:
                    history_models[m_name].append(current_model_s[m_name])
                
                print(f"[Exchange #{total_exchanges}] Iter {iteration} | Pixel (r={r}, c={c}) {bin_a} -> {best_target_bin} | Net: +{best_delta_net:.4f} (d_S: {best_delta_s_total:+.4f}, Pressure: {best_pressure_reward:+.4f})")
                changed = True
                moves_made += 1

    print(f"--- Iteration {iteration} Complete: {moves_made} exchanges. Global Aggregate S = {current_agg_s:.4f} ---\n")

print(f"Consensus Automaton converged after {iteration} iterations with {total_exchanges} total universal exchanges.")

# ==============================================================================
# 5. Final Phase 2 Population Printout
# ==============================================================================
print("\n==================================================")
print("Final Phase 2 Superbin Populations (Nominal D0 Only):")
print("==================================================")
bin_pops_dict = {}
for b_id in active_bins:
    mask = (grid_map == b_id)
    pop = np.sum(grid_nplus[mask])
    bin_pops_dict[int(b_id)] = float(pop)
    print(f"Superbin {b_id:2d}: {pop:9,.0f} events")
print("==================================================\n")

# ==============================================================================
# 5.5 Export Data for Global Diagnostics
# ==============================================================================
summary_data = {
    "target_seeds": TARGET_SEEDS,
    "active_superbins": len(active_bins),
    "target_events": TARGET_EVENTS,
    "final_aggregate_significance": float(current_agg_s),
    "bin_populations": bin_pops_dict,
    "model_significances": {m: float(current_model_s[m]) for m in model_names}
}

summary_json_path = os.path.join(WORK_DIR, "global_phase2_summary.json")
with open(summary_json_path, "w") as f:
    json.dump(summary_data, f, indent=4)
print(f"Exported JSON diagnostic summary to {summary_json_path}")

# ==============================================================================
# 6. Export Optimized State & Generate Histograms
# ==============================================================================
f_out = ROOT.TFile(os.path.join(WORK_DIR, "global_phase2_optimized_state.root"), "RECREATE")

# Final Universal Mask
h_optimized = h_voronoi_in.Clone("h_optimized")
h_optimized.SetTitle("Global Phase 2: Universal Optimized Superbins;m^{2}(#pi^{+}#pi^{-}) [GeV^{2}/c^{4}];m^{2}(#pi^{+}#pi^{0}) [GeV^{2}/c^{4}]")
h_optimized.Reset()
for r in range(GRID_N):
    for c in range(GRID_N):
        h_optimized.SetBinContent(r + 1, c + 1, grid_map[r, c])
h_optimized.Write("h_optimized")

h_acp_hists = {}
h_raw_hists = {}

for m_name in model_names:
    h_acp = h_voronoi_in.Clone(f"h_acp_{m_name}")
    h_acp.SetTitle(f"Phase 2: Superbin Averaged A_{{cp}} ({m_name});m^{{2}}(#pi^{{+}}#pi^{{-}});m^{{2}}(#pi^{{+}}#pi^{{0}})")
    h_acp.Reset()
    
    h_raw = h_voronoi_in.Clone(f"h_raw_acp_{m_name}")
    h_raw.SetTitle(f"Phase 2: Raw Grid A_{{cp}} with Master Boundaries ({m_name});m^{{2}}(#pi^{{+}}#pi^{{-}});m^{{2}}(#pi^{{+}}#pi^{{0}})")
    h_raw.Reset()
    
    grid_nm = grid_nminus_dict[m_name]
    
    for r in range(GRID_N):
        for c in range(GRID_N):
            b_id = grid_map[r, c]
            
            # Raw A_cp
            cell_p = grid_nplus[r, c]
            cell_m = grid_nm[r, c]
            tot_cell = cell_p + cell_m
            h_raw.SetBinContent(r + 1, c + 1, (cell_p - cell_m) / tot_cell if tot_cell > 0 else 0.0)
            
            # Bulk Averaged A_cp
            if b_id > 0:
                mask = (grid_map == b_id)
                bin_p = np.sum(grid_nplus[mask])
                bin_m = np.sum(grid_nm[mask])
                tot_bin = bin_p + bin_m
                h_acp.SetBinContent(r + 1, c + 1, (bin_p - bin_m) / tot_bin if tot_bin > 0 else 0.0)
            else:
                h_acp.SetBinContent(r + 1, c + 1, 0.0)
                
    h_acp.Write(f"h_acp_{m_name}")
    h_raw.Write(f"h_raw_acp_{m_name}")
    h_acp_hists[m_name] = h_acp
    h_raw_hists[m_name] = h_raw

# Build TGraphs (Tracking ONLY the pure Significance, not Net Score)
x_steps = np.array(exchange_steps, dtype=float)
g_agg = ROOT.TGraph(len(exchange_steps), x_steps, np.array(history_aggregate, dtype=float))
g_agg.SetTitle("Aggregate Global Significance (Sum of all models);Exchange Number;Sum of Significance")
g_agg.SetLineColor(ROOT.kBlack)
g_agg.SetLineWidth(3)
g_agg.Write("g_sig_aggregate")

g_models = {}
colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange+7, ROOT.kViolet, ROOT.kTeal+3]
for idx, m_name in enumerate(model_names):
    g = ROOT.TGraph(len(exchange_steps), x_steps, np.array(history_models[m_name], dtype=float))
    g.SetTitle(f"Significance Evolution ({m_name});Exchange Number;Significance S")
    g.SetLineColor(colors[idx % len(colors)])
    g.SetLineWidth(2)
    g.Write(f"g_sig_{m_name}")
    g_models[m_name] = g

# ==============================================================================
# 7. Exact Grid-Edge Tracing for Overlays
# ==============================================================================
boundary_lines = []
xaxis = h_optimized.GetXaxis()
yaxis = h_optimized.GetYaxis()

for r in range(GRID_N):
    for c in range(GRID_N):
        b_current = grid_map[r, c]
        if r + 1 < GRID_N:
            b_right = grid_map[r + 1, c]
            if b_current != b_right and (b_current > 0 or b_right > 0):
                x_e = xaxis.GetBinUpEdge(r + 1)
                y_l = yaxis.GetBinLowEdge(c + 1)
                y_u = yaxis.GetBinUpEdge(c + 1)
                line = ROOT.TLine(x_e, y_l, x_e, y_u)
                line.SetLineColor(ROOT.kBlack)
                line.SetLineWidth(2)
                boundary_lines.append(line)
                
        if c + 1 < GRID_N:
            b_top = grid_map[r, c + 1]
            if b_current != b_top and (b_current > 0 or b_top > 0):
                y_e = yaxis.GetBinUpEdge(c + 1)
                x_l = xaxis.GetBinLowEdge(r + 1)
                x_u = xaxis.GetBinUpEdge(r + 1)
                line = ROOT.TLine(x_l, y_e, x_u, y_e)
                line.SetLineColor(ROOT.kBlack)
                line.SetLineWidth(2)
                boundary_lines.append(line)

# ==============================================================================
# 8. Generate All PNG Visualizations
# ==============================================================================
# Setup Seeds Overlay
g_seeds = ROOT.TGraph()
g_seeds.SetMarkerStyle(29)
g_seeds.SetMarkerSize(2.5)
g_seeds.SetMarkerColor(ROOT.kWhite)
for idx, s in enumerate(seeds):
    g_seeds.SetPoint(idx, s['x'], s['y'])

# Plot 1: Universal Optimized Superbin Map
base_colors = [ROOT.kWhite, ROOT.kBlack]
palette = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+2, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange+7, ROOT.kSpring+4, ROOT.kViolet, ROOT.kAzure+1, ROOT.kTeal+3, ROOT.kPink+2, ROOT.kYellow+2, ROOT.kRed-4, ROOT.kBlue-4, ROOT.kGreen-3, ROOT.kMagenta-3, ROOT.kCyan-3, ROOT.kOrange-3, ROOT.kSpring-5, ROOT.kViolet-6, ROOT.kRed-7, ROOT.kBlue-7, ROOT.kGreen-7, ROOT.kMagenta-7, ROOT.kCyan-7, ROOT.kOrange-7, ROOT.kSpring-7, ROOT.kViolet-7, ROOT.kAzure+6, ROOT.kTeal-5, ROOT.kPink-5, ROOT.kYellow-5]
final_colors = base_colors + palette
custom_colors = array.array('i', final_colors[:TARGET_SEEDS + 2])
ROOT.gStyle.SetPalette(len(custom_colors), custom_colors)

c1 = ROOT.TCanvas("c_opt", "", 800, 800)
c1.SetLeftMargin(0.15)
c1.SetRightMargin(0.18)
h_optimized.SetMinimum(-1)
h_optimized.SetMaximum(TARGET_SEEDS)
h_optimized.SetContour(TARGET_SEEDS + 2)
h_optimized.Draw("COLZ")
g_seeds.Draw("P SAME")
c1.SaveAs(os.path.join(WORK_DIR, "global_phase2_optimized_binning.png"))

# Switch to Temperature Map for ACP Plots
ROOT.gStyle.SetPalette(ROOT.kTemperatureMap)

# Plots 2-17: 8 Bulk ACP Maps + 8 Raw ACP Maps
for m_name in model_names:
    c_acp = ROOT.TCanvas(f"c_acp_{m_name}", "", 800, 800)
    c_acp.SetLeftMargin(0.15)
    c_acp.SetRightMargin(0.18)
    h_acp_hists[m_name].SetMinimum(-0.03)
    h_acp_hists[m_name].SetMaximum(0.03)
    h_acp_hists[m_name].Draw("COLZ")
    for line in boundary_lines: line.Draw("SAME")
    c_acp.SaveAs(os.path.join(WORK_DIR, f"global_phase2_acp_map_{m_name}.png"))
    
    c_raw = ROOT.TCanvas(f"c_raw_{m_name}", "", 800, 800)
    c_raw.SetLeftMargin(0.15)
    c_raw.SetRightMargin(0.18)
    h_raw_hists[m_name].SetMinimum(-0.1)
    h_raw_hists[m_name].SetMaximum(0.1)
    h_raw_hists[m_name].Draw("COLZ")
    for line in boundary_lines: line.Draw("SAME")
    c_raw.SaveAs(os.path.join(WORK_DIR, f"global_phase2_raw_acp_map_{m_name}.png"))

# Plots 18-26: Significance Evolution
c_sig = ROOT.TCanvas("c_sig", "", 1000, 600)
c_sig.SetGrid()
c_sig.SetTickx(1); c_sig.SetTicky(1)
c_sig.SetLeftMargin(0.12); c_sig.SetBottomMargin(0.14)

g_agg.Draw("APL")
c_sig.SaveAs(os.path.join(WORK_DIR, "global_phase2_sig_evol_aggregate.png"))

for m_name in model_names:
    c_sig.Clear()
    g_models[m_name].Draw("APL")
    c_sig.SaveAs(os.path.join(WORK_DIR, f"global_phase2_sig_evol_{m_name}.png"))
f_out.Close()
print(f"\nAll operations complete. Outputs saved to {WORK_DIR}/")

# Prevent PyROOT garbage collection segfault on exit
import os
os._exit(0)