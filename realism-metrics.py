"""
Compute realism metrics for a radial distribution feeder.
Just run this script in the same folder as:
  • BusCoords.csv
  • lines.dss
  • loads.dss
  • sources.dss

Outputs go to ./_metrics/
"""
import os, re, math
import numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# -------------------------
# File paths (edit if needed)
# -------------------------
BUSCOORDS = "BusCoords.csv"
LINES     = "lines.dss"
LOADS     = "loads.dss"
SOURCES   = "sources.dss"
OUT_DIR   = "synth_metrics"
REACH_BIN = 100.0   # meters per bin for reach histogram
# -------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Helper functions ----------
def parse_dss_keyvals(s):
    s = re.split(r"//", s)[0]
    tokens = re.findall(r'([A-Za-z_][A-Za-z0-9_]*?)\s*=\s*("[^"]*"|\S+)', s)
    out = {}
    for k, v in tokens:
        out[k.lower()] = v.strip('"')
    return out

def parse_buscoords(path):
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :3]
    df.columns = ["bus", "x", "y"]
    df["bus"] = df["bus"].astype(str).str.strip()
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna()

def parse_lines(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower().startswith(("!", "c ", "rem ")):
                continue
            if "new" in line.lower() and "line." in line.lower():
                kv = parse_dss_keyvals(line)
                b1 = (kv.get("bus1") or kv.get("bus") or "").split(".")[0]
                b2 = (kv.get("bus2") or "").split(".")[0]
                if not b1 or not b2:
                    continue
                length = kv.get("length")
                units  = kv.get("units", "").lower()
                rows.append({
                    "bus1": b1, "bus2": b2,
                    "length": float(length) if length and re.match(r"^[0-9.]+$", length) else np.nan,
                    "units": units
                })
    return pd.DataFrame(rows)

def parse_sources(path):
    roots = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower().startswith(("!", "c ", "rem ")):
                continue
            if "new" in line.lower() and any(x in line.lower() for x in ["vsource.", "source.", "circuit."]):
                kv = parse_dss_keyvals(line)
                b = (kv.get("bus1") or kv.get("bus") or "").split(".")[0]
                if b: roots.add(b)
    return sorted(roots)

def units_to_meters(length, units):
    if pd.isna(length): return np.nan
    u = (units or "").lower()
    if u in ("m", "meter", "meters", ""): return length
    if u in ("ft", "feet"): return length * 0.3048
    if u in ("mi", "mile", "miles"): return length * 1609.34
    if u in ("km", "kilometer", "kilometers"): return length * 1000
    if u == "kft": return length * 304.8
    return length

def edge_lengths_from_coords(df, coords):
    L = []
    for _, r in df.iterrows():
        b1, b2 = r["bus1"], r["bus2"]
        if b1 in coords and b2 in coords:
            (x1, y1), (x2, y2) = coords[b1], coords[b2]
            L.append(math.hypot(x2 - x1, y2 - y1))
        else:
            L.append(np.nan)
    return np.array(L)

def compute_motifs(G):
    from math import comb
    deg = dict(G.degree())
    star = sum(comb(d, 3) for d in deg.values() if d >= 3)
    path4 = sum(max(deg[u]-1,0)*max(deg[v]-1,0) for u,v in G.edges())
    return pd.DataFrame({
        "motif": ["four_node_star_K1_3", "four_node_path_P4"],
        "count": [int(star), int(path4)]
    })

# ---------- Load data ----------
bus = parse_buscoords(BUSCOORDS)
lines = parse_lines(LINES)
roots = parse_sources(SOURCES)
coords = {r.bus: (r.x, r.y) for _, r in bus.iterrows()}

lines["len_m_dss"] = [units_to_meters(l, u) for l,u in zip(lines["length"], lines["units"])]
lines["len_m_geo"] = edge_lengths_from_coords(lines, coords)
lines["len_m"] = lines["len_m_dss"].where(~lines["len_m_dss"].isna(), lines["len_m_geo"])

# ---------- Build graph ----------
G = nx.Graph()
for b,(x,y) in coords.items():
    G.add_node(b, x=x, y=y)
for _,r in lines.iterrows():
    if r.bus1==r.bus2: continue
    G.add_edge(r.bus1, r.bus2, length_m=float(r.len_m) if not pd.isna(r.len_m) else 1.0)

roots = [r for r in roots if r in G.nodes]
if not roots:
    roots = [max(G.degree, key=lambda kv: kv[1])[0]]

# ---------- Metrics ----------
deg_series = pd.Series(dict(G.degree()))
deg_df = deg_series.value_counts().sort_index().rename_axis("degree").reset_index(name="count")
deg_df["pmf"] = deg_df["count"]/deg_df["count"].sum()

# hops
hop = {}
for n in G.nodes:
    best = np.inf
    for r in roots:
        if nx.has_path(G,r,n):
            d = nx.shortest_path_length(G,r,n)
            best = min(best,d)
    hop[n] = best if best<np.inf else np.nan
hop_s = pd.Series(hop).dropna().astype(int)
hop_df = hop_s.value_counts().sort_index().rename_axis("hops").reset_index(name="count")
hop_df["pmf"] = hop_df["count"]/hop_df["count"].sum()

# reach
reach = {}
for n in G.nodes:
    best = np.inf
    for r in roots:
        if nx.has_path(G,r,n):
            d = nx.shortest_path_length(G,r,n,weight="length_m")
            best = min(best,d)
    reach[n] = best if best<np.inf else np.nan
reach_s = pd.Series(reach).dropna()
edges = np.arange(0, math.ceil(reach_s.max()/REACH_BIN+1)*REACH_BIN, REACH_BIN)
hist,e = np.histogram(reach_s, bins=edges)
reach_df = pd.DataFrame({"reach_bin_m":(e[:-1]+e[1:])/2,"count":hist})
reach_df["pmf"] = reach_df["count"]/reach_df["count"].sum()

motif_df = compute_motifs(G)
summary = {
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "avg_degree": deg_series.mean(),
    "n_roots": len(roots),
    "root_names": ",".join(roots),
    "total_line_length_km": np.nansum([d.get("length_m",0) for _,_,d in G.edges(data=True)])/1000,
    "mean_hop": hop_s.mean(),
    "p90_hop": hop_s.quantile(0.9),
    "mean_reach_m": reach_s.mean(),
    "p90_reach_m": reach_s.quantile(0.9)
}
summary_df = pd.DataFrame([summary])

# ---------- Save outputs ----------
deg_df.to_csv(f"{OUT_DIR}/degree_distribution.csv", index=False)
hop_df.to_csv(f"{OUT_DIR}/hop_distribution.csv", index=False)
reach_df.to_csv(f"{OUT_DIR}/reach_distribution.csv", index=False)
motif_df.to_csv(f"{OUT_DIR}/motif_counts.csv", index=False)
summary_df.to_csv(f"{OUT_DIR}/network_summary.csv", index=False)
pd.DataFrame({"bus":list(G.nodes),"hop":hop_s,"reach_m":reach_s}).to_csv(f"{OUT_DIR}/per_node_metrics.csv")

# ---------- Plots ----------
plt.figure(); plt.bar(deg_df.degree, deg_df.pmf); plt.xlabel("Degree"); plt.ylabel("PMF"); plt.title("Degree distribution")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/degree_distribution.png",dpi=150); plt.close()

plt.figure(); plt.bar(hop_df.hops, hop_df.pmf); plt.xlabel("Hops from substation"); plt.ylabel("PMF"); plt.title("Hop distribution")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/hop_distribution.png",dpi=150); plt.close()

plt.figure(); plt.bar(reach_df.reach_bin_m, reach_df.pmf, width=REACH_BIN*0.9)
plt.xlabel("Reach (meters, bin mid)"); plt.ylabel("PMF"); plt.title("Reach distribution")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/reach_distribution.png",dpi=150); plt.close()

print("All realism metrics generated in:", os.path.abspath(OUT_DIR))
