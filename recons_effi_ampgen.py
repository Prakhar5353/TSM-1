import os
import glob
import ROOT
import numpy as np
import array

# ==============================================================================
# 0. ROOT Configuration
# ==============================================================================

ROOT.ROOT.EnableImplicitMT()
ROOT.gStyle.SetOptStat(0)


# ==============================================================================
# 1. Physical Phase Space Limits & Masking Rules
# ==============================================================================

X_MIN, X_MAX = 0.077, 2.99
Y_MIN, Y_MAX = 0.075, 2.97
KINEMATIC_LIMIT = 3.48
MARGIN = 0.1


def is_physical(x_c, y_c):
    return (X_MIN + MARGIN < x_c < X_MAX - MARGIN) and (Y_MIN + MARGIN < y_c < Y_MAX - MARGIN) and (x_c + y_c < KINEMATIC_LIMIT - MARGIN)


def mask_unphysical(h):
    """Zero out every bin whose centre lies outside the Dalitz boundary."""
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    n_masked = 0

    for ix in range(1, nx + 1):
        x_c = h.GetXaxis().GetBinCenter(ix)
        for iy in range(1, ny + 1):
            y_c = h.GetYaxis().GetBinCenter(iy)
            if not is_physical(x_c, y_c):
                h.SetBinContent(ix, iy, 0.0)
                h.SetBinError(ix, iy, 0.0)
                n_masked += 1

    return n_masked


# ==============================================================================
# 2. Reconstructed MC Kinematics
# ==============================================================================

def add_reco_kinematics(df):
    # Invariant mass of pi+ pi-
    df = df.Define("PX_pippim", "H1_PX + H2_PX").Define("PY_pippim", "H1_PY + H2_PY").Define("PZ_pippim", "H1_PZ + H2_PZ").Define("PE_pippim", "H1_ENERGY + H2_ENERGY")
    df = df.Define("m2_pippim", "(PE_pippim*PE_pippim - (PX_pippim*PX_pippim + PY_pippim*PY_pippim + PZ_pippim*PZ_pippim)) * 1e-6")

    # Identify the true positive pion
    df = df.Define("pip_PX", "H1_TRUEID == 211 ? H1_PX : H2_PX").Define("pip_PY", "H1_TRUEID == 211 ? H1_PY : H2_PY").Define("pip_PZ", "H1_TRUEID == 211 ? H1_PZ : H2_PZ").Define("pip_PE", "H1_TRUEID == 211 ? H1_ENERGY : H2_ENERGY")

    # Invariant mass of pi+ pi0
    df = df.Define("PX_pippi0", "pip_PX + pi0_PX").Define("PY_pippi0", "pip_PY + pi0_PY").Define("PZ_pippi0", "pip_PZ + pi0_PZ").Define("PE_pippi0", "pip_PE + pi0_ENERGY")
    df = df.Define("m2_pippi0", "(PE_pippi0*PE_pippi0 - (PX_pippi0*PX_pippi0 + PY_pippi0*PY_pippi0 + PZ_pippi0*PZ_pippi0)) * 1e-6")

    return df


# ==============================================================================
# 3. Main Function
# ==============================================================================

def main():
    # Output directory and filenames
    out_dir = "efficiency"
    os.makedirs(out_dir, exist_ok=True)

    out_img = os.path.join(out_dir, "dalitz_efficiency_map_ampgen_denom.png")
    out_root = os.path.join(out_dir, "dalitz_efficiency_map_ampgen_denom.root")
    out_ampgen = os.path.join(out_dir, "ampgen_dalitz.png")
    out_reco = os.path.join(out_dir, "MC_recons_dalitz.png")

    # Variable-width binning
    m2_rho770 = 0.77477**2
    m2_f21270 = 1.275**2
    m2_rho1450 = 1.44502**2

    anchors = [0.0, m2_rho770, m2_f21270, m2_rho1450, 3.1]
    n_total_bins = 50
    lengths = np.diff(anchors)
    total_length = 3.1

    # Allocate bins using the largest-remainder method
    raw_bins = lengths / total_length * n_total_bins
    bins_per_segment = np.floor(raw_bins).astype(int)
    remaining = n_total_bins - bins_per_segment.sum()
    remainders = raw_bins - bins_per_segment

    for i in np.argsort(remainders)[::-1][:remaining]:
        bins_per_segment[i] += 1

    print("\n--- Variable-width binning ---")
    print("Bins per segment:", bins_per_segment)
    print("Total bins:", bins_per_segment.sum())

    # Construct bin edges
    edges = [0.0]
    for i in range(len(anchors) - 1):
        seg_edges = np.linspace(anchors[i], anchors[i + 1], bins_per_segment[i] + 1)
        edges.extend(seg_edges[1:])

    bin_edges_x = array.array('d', edges)
    bin_edges_y = array.array('d', edges)

    th2_model = ROOT.RDF.TH2DModel("h_denominator_mc", "Denominator", n_total_bins, bin_edges_x, n_total_bins, bin_edges_y)

    # ==========================================================================
    # 4. Process Denominator: AmpGen
    # ==========================================================================

    denom_file = "efficiency_map_denom_ampgen.root"
    if not os.path.exists(denom_file):
        raise FileNotFoundError(f"AmpGen denominator file not found: {denom_file}")

    print("\nLoading AmpGen denominator...")
    df_den = ROOT.RDataFrame("DalitzEventList", denom_file)

    # pi+ pi- invariant mass
    df_den = df_den.Define("PE_pippim", "pip_E + pim_E").Define("PX_pippim", "pip_Px + pim_Px").Define("PY_pippim", "pip_Py + pim_Py").Define("PZ_pippim", "pip_Pz + pim_Pz")
    df_den = df_den.Define("m2_pippim", "PE_pippim*PE_pippim - (PX_pippim*PX_pippim + PY_pippim*PY_pippim + PZ_pippim*PZ_pippim)")

    # pi+ pi0 invariant mass
    df_den = df_den.Define("PE_pippi0", "pip_E + pi0_E").Define("PX_pippi0", "pip_Px + pi0_Px").Define("PY_pippi0", "pip_Py + pi0_Py").Define("PZ_pippi0", "pip_Pz + pi0_Pz")
    df_den = df_den.Define("m2_pippi0", "PE_pippi0*PE_pippi0 - (PX_pippi0*PX_pippi0 + PY_pippi0*PY_pippi0 + PZ_pippi0*PZ_pippi0)")

    # Unweighted denominator: every AmpGen event contributes 1
    h_den = df_den.Histo2D(th2_model, "m2_pippim", "m2_pippi0").GetValue()
    print(f"AmpGen denominator entries: {h_den.GetEntries():,.0f}")

    # ==========================================================================
    # 5. Process Numerator: LHCb Reconstructed MC
    # ==========================================================================

    base_dir = "/panfs/avilla/D02pipipi0/d0_to_hhpi0_mc"
    files_40_41 = glob.glob(f"{base_dir}/d0_to_hhpi0_mc_2024_w40_41_magdown/*.root")
    files_41_42 = glob.glob(f"{base_dir}/d0_to_hhpi0_mc_2024_w41_42_magup/*.root")
    file_list = sorted(files_40_41 + files_41_42)

    if not file_list:
        raise FileNotFoundError(f"No ROOT files found under {base_dir}")

    print(f"\nNumber of reconstructed MC files: {len(file_list)}")

    chain_r = ROOT.TChain("Dst2D0pi_D02pipipi0_R/DecayTree")
    chain_m = ROOT.TChain("Dst2D0pi_D02pipipi0_M/DecayTree")

    for f in file_list:
        chain_r.Add(f)
        chain_m.Add(f)

    if chain_r.GetEntries() == 0:
        raise RuntimeError("No entries found in the R tree.")
    if chain_m.GetEntries() == 0:
        raise RuntimeError("No entries found in the M tree.")

    print(f"R tree entries: {chain_r.GetEntries():,}")
    print(f"M tree entries: {chain_m.GetEntries():,}")

    df_r = ROOT.RDataFrame(chain_r)
    df_m = ROOT.RDataFrame(chain_m)

    # Signal truth matching & reconstruction selection
    signal = "D0_TRUEID == 421 && abs(H1_TRUEID) == 211 && abs(H2_TRUEID) == 211 && H1_TRUEID * H2_TRUEID < 0 && pi0_TRUEID == 111"
    reco_cuts = f"{signal} && (Dst_Hlt1TrackMVADecision_TOS == 1 || Dst_Hlt1TwoTrackMVADecision_TOS == 1) && D0_OWNPVDIRA > 0.9995 && D0_OWNPVIPCHI2 > 0 && D0_OWNPVIPCHI2 < 30 && Dst_MAXDOCACHI2 < 20 && H1_TRGHOSTPROB < 0.35 && H2_TRGHOSTPROB < 0.35 && Dst_DTF_PV_Mpi0_DTFCHI2 < 30"

    df_r = df_r.Filter(reco_cuts)
    df_m = df_m.Filter(reco_cuts)

    # Replace reconstructed kinematics with TRUE kinematics
    for prefix in ["H1", "H2", "pi0"]:
        df_r = df_r.Redefine(f"{prefix}_PX", f"{prefix}_TRUEPX").Redefine(f"{prefix}_PY", f"{prefix}_TRUEPY").Redefine(f"{prefix}_PZ", f"{prefix}_TRUEPZ").Redefine(f"{prefix}_E", f"{prefix}_TRUEENERGY")
        df_m = df_m.Redefine(f"{prefix}_PX", f"{prefix}_TRUEPX").Redefine(f"{prefix}_PY", f"{prefix}_TRUEPY").Redefine(f"{prefix}_PZ", f"{prefix}_TRUEPZ").Redefine(f"{prefix}_E", f"{prefix}_TRUEENERGY")

    df_r = df_r.Alias("H1_ENERGY", "H1_E").Alias("H2_ENERGY", "H2_E").Alias("pi0_ENERGY", "pi0_E")
    df_m = df_m.Alias("H1_ENERGY", "H1_E").Alias("H2_ENERGY", "H2_E").Alias("pi0_ENERGY", "pi0_E")

    df_r = add_reco_kinematics(df_r)
    df_m = add_reco_kinematics(df_m)

    # Extract selected columns
    cols = ["RUNNUMBER", "EVENTNUMBER", "m2_pippim", "m2_pippi0"]
    print("\nExtracting reconstructed MC events...")

    arr_r = df_r.AsNumpy(cols)
    arr_m = df_m.AsNumpy(cols)

    print(f"R candidates after cuts: {len(arr_r['RUNNUMBER']):,}")
    print(f"M candidates after cuts: {len(arr_m['RUNNUMBER']):,}")

    # Combine R and M samples
    combined_run = np.concatenate([arr_r["RUNNUMBER"], arr_m["RUNNUMBER"]]).astype(np.int64)
    combined_event = np.concatenate([arr_r["EVENTNUMBER"], arr_m["EVENTNUMBER"]]).astype(np.int64)
    combined_x = np.concatenate([arr_r["m2_pippim"], arr_m["m2_pippim"]]).astype(np.float64)
    combined_y = np.concatenate([arr_r["m2_pippi0"], arr_m["m2_pippi0"]]).astype(np.float64)

    combined_data = np.empty(len(combined_run), dtype=[("run", np.int64), ("event", np.int64), ("x", np.float64), ("y", np.float64)])
    combined_data["run"] = combined_run
    combined_data["event"] = combined_event
    combined_data["x"] = combined_x
    combined_data["y"] = combined_y

    # Remove exact duplicate rows
    data = np.unique(combined_data)

    print("\n--- Reconstructed MC statistics ---")
    print(f"Total candidates before deduplication: {len(combined_data):,}")
    print(f"Candidates after deduplication: {len(data):,}")
    print(f"Duplicate rows removed: {len(combined_data) - len(data):,}")

    # Fill numerator histogram
    h_num = ROOT.TH2D("h_numerator_reco", "Numerator", n_total_bins, bin_edges_x, n_total_bins, bin_edges_y)
    for event in data:
        h_num.Fill(event["x"], event["y"])

    print(f"Reconstructed numerator entries: {h_num.GetEntries():,.0f}")

    # ==========================================================================
    # 6. Mask Unphysical Bins and Clear Flow Bins
    # ==========================================================================

    n1 = mask_unphysical(h_den)
    n2 = mask_unphysical(h_num)

    print("\n--- Dalitz masking ---")
    print(f"Masked AmpGen bins: {n1}")
    print(f"Masked reconstructed bins: {n2}")

    nx, ny = h_num.GetNbinsX(), h_num.GetNbinsY()
    for ix in range(0, nx + 2):
        for iy in range(0, ny + 2):
            if ix == 0 or ix == nx + 1 or iy == 0 or iy == ny + 1:
                h_num.SetBinContent(ix, iy, 0.0)
                h_num.SetBinError(ix, iy, 0.0)
                h_den.SetBinContent(ix, iy, 0.0)
                h_den.SetBinError(ix, iy, 0.0)

    # ==========================================================================
    # 7. Plot Individual Dalitz Distributions
    # ==========================================================================

    ROOT.gStyle.SetPalette(ROOT.kBird)

    c_ampgen = ROOT.TCanvas("c_ampgen", "AmpGen Dalitz Distribution", 800, 600)
    c_ampgen.SetRightMargin(0.15)
    c_ampgen.SetLeftMargin(0.12)
    h_den.SetTitle("AmpGen Generated Dalitz Distribution;m^{2}(#pi^{+}#pi^{-}) [GeV^{2}/c^{4}];m^{2}(#pi^{+}#pi^{0}) [GeV^{2}/c^{4}]")
    h_den.Draw("COLZ")
    c_ampgen.SaveAs(out_ampgen)
    print(f"\nSaved AmpGen Dalitz plot to: {out_ampgen}")

    c_reco = ROOT.TCanvas("c_reco", "Reconstructed MC Dalitz Distribution", 800, 600)
    c_reco.SetRightMargin(0.15)
    c_reco.SetLeftMargin(0.12)
    h_num.SetTitle("Reconstructed MC Dalitz Distribution;m^{2}(#pi^{+}#pi^{-}) [GeV^{2}/c^{4}];m^{2}(#pi^{+}#pi^{0}) [GeV^{2}/c^{4}]")
    h_num.Draw("COLZ")
    c_reco.SaveAs(out_reco)
    print(f"Saved reconstructed MC Dalitz plot to: {out_reco}")

    # ==========================================================================
    # 8. Calculate Efficiency Map
    # ==========================================================================

    h_eff_map = h_den.Clone("h_efficiency_map")
    h_eff_map.Reset()
    h_eff_map.SetTitle("Reconstruction Efficiency (#epsilon - AmpGen Denom);m^{2}(#pi^{+}#pi^{-}) [GeV^{2}/c^{4}];m^{2}(#pi^{+}#pi^{0}) [GeV^{2}/c^{4}]")

    for ix in range(1, n_total_bins + 1):
        for iy in range(1, n_total_bins + 1):
            num_val, den_val = h_num.GetBinContent(ix, iy), h_den.GetBinContent(ix, iy)
            efficiency = (num_val / den_val) if den_val > 0 else 0.0
            h_eff_map.SetBinContent(ix, iy, efficiency)
            h_eff_map.SetBinError(ix, iy, 0.0)

    # ==========================================================================
    # 9. Efficiency Validation: Identify Bins with Efficiency > 1
    # ==========================================================================

    print("\n========================================")
    print("       EFFICIENCY VALIDATION")
    print("========================================")

    max_efficiency = h_eff_map.GetMaximum()
    print(f"Maximum efficiency: {max_efficiency:.6f}")

    n_over_one = 0
    for ix in range(1, n_total_bins + 1):
        for iy in range(1, n_total_bins + 1):
            efficiency = h_eff_map.GetBinContent(ix, iy)
            if efficiency > 1.0:
                n_over_one += 1
                ampgen_count, reco_count = h_den.GetBinContent(ix, iy), h_num.GetBinContent(ix, iy)
                x_low, x_high = h_eff_map.GetXaxis().GetBinLowEdge(ix), h_eff_map.GetXaxis().GetBinUpEdge(ix)
                y_low, y_high = h_eff_map.GetYaxis().GetBinLowEdge(iy), h_eff_map.GetYaxis().GetBinUpEdge(iy)
                print(f"\nProblematic bin (ix={ix}, iy={iy}):\n  m2(pip,pim) range = [{x_low:.5f}, {x_high:.5f}] GeV^2\n  m2(pip,pi0) range = [{y_low:.5f}, {y_high:.5f}] GeV^2\n  AmpGen denominator = {ampgen_count:.0f}\n  Reconstructed MC   = {reco_count:.0f}\n  Efficiency         = {efficiency:.6f}")

    print("\n----------------------------------------")
    print(f"Total bins with efficiency > 1: {n_over_one}")
    print("----------------------------------------")

    # ==========================================================================
    # 10. Dynamic Z-axis Scaling & 11. Plot Efficiency Map
    # ==========================================================================

    z_min = 0.0
    z_max = h_eff_map.GetMaximum()

    print("\n--- Dynamic Z-Axis Scaling ---")
    print(f"Efficiency Map Min: {z_min}")
    print(f"Efficiency Map Max: {z_max:.6f}")

    ROOT.gStyle.SetPalette(ROOT.kBird)
    c1 = ROOT.TCanvas("c1", "Efficiency Map AmpGen Denom", 800, 600)
    c1.SetRightMargin(0.15)
    c1.SetLeftMargin(0.12)
    h_eff_map.SetMinimum(z_min)
    if z_max > 0:
        h_eff_map.SetMaximum(z_max * 1.05)
    h_eff_map.Draw("COLZ")
    c1.SaveAs(out_img)
    print(f"\nSaved efficiency map to: {out_img}")

    # ==========================================================================
    # 12. Save Histograms to ROOT File
    # ==========================================================================

    out_file = ROOT.TFile(out_root, "RECREATE")
    if not out_file or out_file.IsZombie():
        raise RuntimeError(f"Could not create ROOT output file: {out_root}")

    out_file.cd()
    h_den.Write()
    h_num.Write()
    h_eff_map.Write()
    out_file.Close()
    print(f"Saved ROOT histograms to: {out_root}")

    # ==========================================================================
    # 13. Final Summary
    # ==========================================================================

    print("\n========================================")
    print("             PROCESSING DONE")
    print("========================================")
    print("\nOutput files:")
    print(f"  AmpGen Dalitz:       {out_ampgen}")
    print(f"  Reconstructed MC:    {out_reco}")
    print(f"  Efficiency map:      {out_img}")
    print(f"  ROOT histograms:     {out_root}")
    print("\n========================================")


# ==============================================================================
# 14. Execute
# ==============================================================================

if __name__ == "__main__":
    main()