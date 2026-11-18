import array
import math
import os
import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.EnableImplicitMT()

# ==============================================================================
# 1. Dataset File Paths (Baseline + 4 Decremental (dec) + 4 Incremental (inc))
# ==============================================================================
base_dir = "/home/pbhattac/optimize"

paths = {
    "D0_const":       f"{base_dir}/Dtopipipi0.root",
    
    # Decremental (dec) Control Files
    "rhom_mag":       f"{base_dir}/1sigma/Dbartopipipi0_rhom_mag_dec.root",
    "rhon_mag":       f"{base_dir}/1sigma/Dbartopipipi0_rhon_mag_dec.root",
    "rhom_phase":     f"{base_dir}/1sigma/Dbartopipipi0_rhom_phase_dec.root",
    "rhon_phase":     f"{base_dir}/1sigma/Dbartopipipi0_rhon_phase_dec.root",
    
    # Incremental (inc) Control Files
    "rhom_mag_inc":   f"{base_dir}/1sigma/Dbartopipipi0_rhom_mag_inc.root",
    "rhon_mag_inc":   f"{base_dir}/1sigma/Dbartopipipi0_rhon_mag_inc.root",
    "rhom_phase_inc": f"{base_dir}/1sigma/Dbartopipipi0_rhom_phase_inc.root",
    "rhon_phase_inc": f"{base_dir}/1sigma/Dbartopipipi0_rhon_phase_inc.root",
}

# ==============================================================================
# 2. Dynamic RDataFrame Processing Helper (Robust Column Resolution)
# ==============================================================================
def load_and_define_df(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"No read permission for file: {file_path}")

    df = ROOT.RDataFrame("DalitzEventList", file_path)
    cols = set(str(c) for c in df.GetColumnNames())

    # Case A: Standard AmpGen raw branch names (_1_pi~_E, etc.)
    if "_1_pi~_E" in cols:
        df = df.Alias("pim_E",  "_1_pi~_E").Alias("pim_PX", "_1_pi~_Px").Alias("pim_PY", "_1_pi~_Py").Alias("pim_PZ", "_1_pi~_Pz")
        df = df.Alias("pip_E",  "_2_pi#_E").Alias("pip_PX", "_2_pi#_Px").Alias("pip_PY", "_2_pi#_Py").Alias("pip_PZ", "_2_pi#_Pz")
        df = df.Alias("pi0_E",  "_3_pi0_E").Alias("pi0_PX", "_3_pi0_Px").Alias("pi0_PY", "_3_pi0_Py").Alias("pi0_PZ", "_3_pi0_Pz")

    # Case B: Clean branch names (pim_Px -> pim_PX)
    else:
        mappings = [
            ("pim_PX", "pim_Px"), ("pim_PY", "pim_Py"), ("pim_PZ", "pim_Pz"),
            ("pip_PX", "pip_Px"), ("pip_PY", "pip_Py"), ("pip_PZ", "pip_Pz"),
            ("pi0_PX", "pi0_Px"), ("pi0_PY", "pi0_Py"), ("pi0_PZ", "pi0_Pz"),
        ]
        for target, source in mappings:
            if target not in cols and source in cols:
                df = df.Alias(target, source)

    # Invariant mass squared calculations mapped to GeV^2/c^4 (no multiplier needed for AmpGen)
    df = df.Define("m2_pippim", "(pow(pip_E+pim_E,2) - pow(pip_PX+pim_PX,2) - pow(pip_PY+pim_PY,2) - pow(pip_PZ+pim_PZ,2))")
    df = df.Define("m2_pimpi0", "(pow(pim_E+pi0_E,2) - pow(pim_PX+pi0_PX,2) - pow(pim_PY+pi0_PY,2) - pow(pim_PZ+pi0_PZ,2))")
    df = df.Define("m2_pippi0", "(pow(pip_E+pi0_E,2) - pow(pip_PX+pi0_PX,2) - pow(pip_PY+pi0_PY,2) - pow(pip_PZ+pi0_PZ,2))")

    total_events = df.Sum("weight").GetValue() if "weight" in cols else df.Count().GetValue()
    return df, total_events

# Load DataFrames and total event counts for all files
dfs, totals = {}, {}
for key, path in paths.items():
    dfs[key], totals[key] = load_and_define_df(path)

# ==============================================================================
# 3. Resonance Definition & Explicit Binning (GeV^2/c^4)
# ==============================================================================
m2_rho770  = 0.77477**2
m2_f21270  = 1.275**2
m2_rho1450 = 1.44502**2

bin_edges = array.array('d', [
    0.0, 0.30, 0.45, m2_rho770, 0.75, 0.90,
    1.25, m2_f21270, 1.85, m2_rho1450, 2.60, 3.10
])
n_bins = len(bin_edges) - 1

resonances = [
    {"name": "#rho(770)",    "m2": m2_rho770,  "color": ROOT.kRed,   "has_horizontal": True},
    {"name": "f_{2}(1270)",  "m2": m2_f21270,  "color": ROOT.kWhite, "has_horizontal": False},
    {"name": "#rho(1450)",   "m2": m2_rho1450, "color": ROOT.kCyan,  "has_horizontal": True},
]

# ==============================================================================
# 4. Fill 2D Histograms for Baseline and Control Files (No Axis Flipping)
# ==============================================================================
N_tot_D0 = totals["D0_const"]
h_D0_cp = dfs["D0_const"].Histo2D(
    ("h_D0_cp", "D0", n_bins, bin_edges, n_bins, bin_edges),
    "m2_pippim", "m2_pippi0", "weight"
).GetValue()

h_sample_cps = {}
all_samples = [
    "rhom_mag", "rhon_mag", "rhom_phase", "rhon_phase",
    "rhom_mag_inc", "rhon_mag_inc", "rhom_phase_inc", "rhon_phase_inc"
]

for key in all_samples:
    h_sample_cps[key] = dfs[key].Histo2D(
        (f"h_{key}_cp", key, n_bins, bin_edges, n_bins, bin_edges),
        "m2_pippim", "m2_pippi0", "weight"
    ).GetValue()

# ==============================================================================
# 5. Resonance Overlay Helper Function
# ==============================================================================
def draw_resonance_lines(legend):
    lines = []
    for res in resonances:
        m2 = res["m2"]
        color = res["color"]
        
        # Vertical line for all resonances on m2(pi+ pi-)
        v_line = ROOT.TLine(m2, 0.0, m2, 3.10)
        v_line.SetLineColor(color)
        v_line.SetLineWidth(3)
        v_line.SetLineStyle(2)
        v_line.Draw("SAME")
        lines.append(v_line)
        
        # Horizontal line drawn only for charged resonance states (rho+, rho-)
        if res["has_horizontal"]:
            h_line = ROOT.TLine(0.0, m2, 3.10, m2)
            h_line.SetLineColor(color)
            h_line.SetLineWidth(3)
            h_line.SetLineStyle(2)
            h_line.Draw("SAME")
            lines.append(h_line)
        
        legend.AddEntry(v_line, res["name"], "l")
    return lines

# ==============================================================================
# 6. Define Comparative Pair Groups (dec vs. inc)
# ==============================================================================
comparison_pairs = [
    {
        "pair_id": "rhom_mag",
        "dec_key": "rhom_mag",
        "inc_key": "rhom_mag_inc",
        "title_label": "#rho(770) mag_diff (m)"
    },
    {
        "pair_id": "rhon_mag",
        "dec_key": "rhon_mag",
        "inc_key": "rhon_mag_inc",
        "title_label": "#rho(770) mag_diff (n)"
    },
    {
        "pair_id": "rhom_phase",
        "dec_key": "rhom_phase",
        "inc_key": "rhom_phase_inc",
        "title_label": "#rho(770) phase_diff (m)"
    },
    {
        "pair_id": "rhon_phase",
        "dec_key": "rhon_phase",
        "inc_key": "rhon_phase_inc",
        "title_label": "#rho(770) phase_diff (n)"
    },
]

# ==============================================================================
# 7. Generate Side-by-Side 2D Asymmetry Comparison Figures (dec vs. inc)
# ==============================================================================
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kTemperatureMap)

os.makedirs("plots", exist_ok=True)

canvas = ROOT.TCanvas("c_comp", "Side-by-Side Asymmetry Comparison (dec vs inc)", 1600, 800)

for pair in comparison_pairs:
    pair_id = pair["pair_id"]
    dec_key = pair["dec_key"]
    inc_key = pair["inc_key"]
    label   = pair["title_label"]

    canvas.Clear()
    canvas.Divide(2, 1)

    def make_asymmetry_hist(key, title_tag):
        hist_bar = h_sample_cps[key]
        N_tot_bar = totals[key]
        
        h_asymm = h_D0_cp.Clone(f"h_asymm_{key}_comp")
        h_asymm.Reset()

        for i in range(1, n_bins + 1):
            for j in range(1, n_bins + 1):
                N_plus = h_D0_cp.GetBinContent(i, j) / N_tot_D0
                N_minus = hist_bar.GetBinContent(i, j) / N_tot_bar
                den_ij = N_plus + N_minus
                if den_ij > 0:
                    h_asymm.SetBinContent(i, j, (N_plus - N_minus) / den_ij)
                else:
                    h_asymm.SetBinContent(i, j, 0)

        title_text = (
            f"{title_tag};"
            f"m^{{2}}(#pi^{{+}}#pi^{{-}}) [GeV^{{2}}/c^{{4}}];"
            f"m^{{2}}(#pi^{{+}}#pi^{{0}}) [GeV^{{2}}/c^{{4}}];"
            f"Asymmetry"
        )
        h_asymm.SetTitle(title_text)
        return h_asymm

    # --- PAD 1: Decremental (dec) Control File ---
    pad1 = canvas.cd(1)
    pad1.SetLeftMargin(0.15)
    pad1.SetRightMargin(0.15)
    
    h_dec_asymm = make_asymmetry_hist(dec_key, f"dec: {label}")
    h_dec_asymm.Draw("COLZ0")
    h_dec_asymm.GetZaxis().SetNoExponent(False)
    h_dec_asymm.GetZaxis().SetMaxDigits(3)
    h_dec_asymm.GetZaxis().SetRangeUser(-0.05, 0.05)
    
    leg1 = ROOT.TLegend(0.55, 0.75, 0.82, 0.88)
    leg1.SetBorderSize(1)
    leg1.SetFillColor(ROOT.kBlack)
    leg1.SetTextColor(ROOT.kWhite)
    leg1.SetTextSize(0.032)
    lines1 = draw_resonance_lines(leg1)
    leg1.Draw("SAME")
    pad1.Update()

    # --- PAD 2: Incremental (inc) Control File ---
    pad2 = canvas.cd(2)
    pad2.SetLeftMargin(0.15)
    pad2.SetRightMargin(0.15)
    
    h_inc_asymm = make_asymmetry_hist(inc_key, f"inc: {label}")
    h_inc_asymm.Draw("COLZ0")
    h_inc_asymm.GetZaxis().SetNoExponent(False)
    h_inc_asymm.GetZaxis().SetMaxDigits(3)
    h_inc_asymm.GetZaxis().SetRangeUser(-0.05, 0.05)
    
    leg2 = ROOT.TLegend(0.55, 0.75, 0.82, 0.88)
    leg2.SetBorderSize(1)
    leg2.SetFillColor(ROOT.kBlack)
    leg2.SetTextColor(ROOT.kWhite)
    leg2.SetTextSize(0.032)
    lines2 = draw_resonance_lines(leg2)
    leg2.Draw("SAME")
    pad2.Update()

    output_png = f"plots/asymmetry_comparison_{pair_id}.png"
    canvas.SaveAs(output_png)
    print(f"Saved side-by-side comparison: {output_png}")

print("\nFinished: All 4 combined side-by-side asymmetry figures saved in 'plots/asymmetry_comparison_*.png'")