#!/usr/bin/env python3
"""
Chapter 5 Report Generator - TikZ Version (v7.4 - FINAL DEBUGGED)
Generates LaTeX-ready TikZ codes + comparison table for 8 figures
Fixed: Proper JSON parsing for pareto data
"""

import pickle
import json
import numpy as np
from pathlib import Path

# ✅ Correct data path
DATA_DIR = Path("analysis/realtime")
OUTPUT_DIR = DATA_DIR / "tikz_codes"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load realtime_cache.pkl and pareto_snapshot.json"""
    try:
        # Load cache
        with open(DATA_DIR / "realtime_cache.pkl", 'rb') as f:
            cache = pickle.load(f)
        
        # Load pareto
        with open(DATA_DIR / "pareto_snapshot.json", 'r') as f:
            pareto_raw = json.load(f)
        
        # DEBUG: Print raw structure
        print(f"🔍 DEBUG: pareto_raw type: {type(pareto_raw)}")
        if isinstance(pareto_raw, list) and len(pareto_raw) > 0:
            print(f"   First element type: {type(pareto_raw[0])}")
            print(f"   First element sample: {pareto_raw[0][:200] if isinstance(pareto_raw[0], str) else pareto_raw[0]}")
        
        # Parse pareto properly
        pareto = []
        if isinstance(pareto_raw, list):
            for item in pareto_raw:
                if isinstance(item, str):
                    # It's a JSON string, parse it
                    try:
                        parsed = json.loads(item)
                        pareto.append(parsed)
                    except json.JSONDecodeError:
                        print(f"⚠️  Failed to parse: {item[:100]}")
                        continue
                elif isinstance(item, dict):
                    # Already a dict
                    pareto.append(item)
                else:
                    print(f"⚠️  Unknown type: {type(item)}")
        elif isinstance(pareto_raw, dict):
            pareto = [pareto_raw]
        else:
            print(f"⚠️  Unexpected pareto structure: {type(pareto_raw)}")
            pareto = []
        
        print(f"✅ Loaded {len(cache['U_history'])} episodes + {len(pareto)} Pareto solutions")
        if len(pareto) > 0:
            print(f"   Sample Pareto point: {pareto[0]}")
        
        return cache, pareto
    
    except FileNotFoundError as e:
        print(f"❌ No data found!")
        print(f"   Looking in: {DATA_DIR.absolute()}")
        print(f"   Error: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_utility_tikz(cache):
    """Generate TikZ code for Utility convergence"""
    utilities = cache['U_history']
    episodes = list(range(len(utilities)))
    
    # Downsample for clarity
    step = max(1, len(episodes) // 100)
    eps_sample = episodes[::step]
    util_sample = utilities[::step]
    
    coords = " ".join([f"({e},{u:.3f})" for e, u in zip(eps_sample, util_sample)])
    
    tikz = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.8\textwidth,
    height=0.5\textwidth,
    xlabel={Episode},
    ylabel={Utility ($U$)},
    grid=major,
    legend pos=south east
]
\addplot[blue, thick] coordinates {
""" + coords + r"""
};
\legend{MATO-UAV}
\end{axis}
\end{tikzpicture}"""
    return tikz

def generate_error_tikz(cache):
    """Generate TikZ code for Error Rate"""
    errors = cache['Delta_history']
    episodes = list(range(len(errors)))
    
    step = max(1, len(episodes) // 100)
    eps_sample = episodes[::step]
    err_sample = errors[::step]
    
    coords = " ".join([f"({e},{err:.4f})" for e, err in zip(eps_sample, err_sample)])
    
    tikz = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.8\textwidth,
    height=0.5\textwidth,
    xlabel={Episode},
    ylabel={Error Rate ($\Delta$)},
    grid=major,
    legend pos=north east
]
\addplot[red, thick] coordinates {
""" + coords + r"""
};
\legend{MATO-UAV}
\end{axis}
\end{tikzpicture}"""
    return tikz

def generate_stability_tikz(cache):
    """Generate TikZ code for Stability (Omega)"""
    stability = cache['Omega_history']
    episodes = list(range(len(stability)))
    
    step = max(1, len(episodes) // 100)
    eps_sample = episodes[::step]
    stab_sample = stability[::step]
    
    coords = " ".join([f"({e},{s:.3f})" for e, s in zip(eps_sample, stab_sample)])
    
    tikz = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.8\textwidth,
    height=0.5\textwidth,
    xlabel={Episode},
    ylabel={Stability ($\Omega$)},
    grid=major,
    legend pos=south east
]
\addplot[green!70!black, thick] coordinates {
""" + coords + r"""
};
\legend{MATO-UAV}
\end{axis}
\end{tikzpicture}"""
    return tikz

def generate_energy_tikz(cache):
    """Generate TikZ code for Energy Consumption"""
    energy = cache['Energy_history']
    episodes = list(range(len(energy)))
    
    step = max(1, len(episodes) // 100)
    eps_sample = episodes[::step]
    energy_sample = energy[::step]
    
    coords = " ".join([f"({e},{en:.2f})" for e, en in zip(eps_sample, energy_sample)])
    
    tikz = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.8\textwidth,
    height=0.5\textwidth,
    xlabel={Episode},
    ylabel={Energy (J)},
    grid=major,
    legend pos=north east
]
\addplot[orange, thick] coordinates {
""" + coords + r"""
};
\legend{MATO-UAV}
\end{axis}
\end{tikzpicture}"""
    return tikz

def generate_delay_tikz(cache):
    """Generate TikZ code for Delay Performance"""
    delay = cache['Delay_history']
    episodes = list(range(len(delay)))
    
    step = max(1, len(episodes) // 100)
    eps_sample = episodes[::step]
    delay_sample = delay[::step]
    
    coords = " ".join([f"({e},{d:.3f})" for e, d in zip(eps_sample, delay_sample)])
    
    tikz = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.8\textwidth,
    height=0.5\textwidth,
    xlabel={Episode},
    ylabel={Delay (s)},
    grid=major,
    legend pos=north east
]
\addplot[purple, thick] coordinates {
""" + coords + r"""
};
\legend{MATO-UAV}
\end{axis}
\end{tikzpicture}"""
    return tikz

def generate_pareto_tikz(pareto):
    """Generate TikZ code for 2D Pareto Front (Utility vs Error)"""
    if not pareto:
        print("⚠️  No pareto data for Pareto plot")
        return "% No Pareto data available"
    
    utilities = []
    errors = []
    
    for p in pareto:
        if not isinstance(p, dict):
            print(f"⚠️  Skipping non-dict pareto point: {type(p)}")
            continue
        
        # Try different key variations
        u = p.get('U', p.get('utility', p.get('Utility', None)))
        d = p.get('Delta', p.get('error_rate', p.get('delta', None)))
        
        if u is not None and d is not None:
            utilities.append(float(u))
            errors.append(float(d))
        else:
            print(f"⚠️  Missing U or Delta in: {p}")
    
    if not utilities:
        return "% No valid Pareto points found"
    
    coords = " ".join([f"({u:.3f},{e:.4f})" for u, e in zip(utilities, errors)])
    
    tikz = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.8\textwidth,
    height=0.6\textwidth,
    xlabel={Utility ($U$)},
    ylabel={Error Rate ($\Delta$)},
    grid=major,
    scatter/classes={a={mark=o,blue}}
]
\addplot[scatter,only marks,mark size=2pt] coordinates {
""" + coords + r"""
};
\end{axis}
\end{tikzpicture}"""
    return tikz

def generate_energy_delay_tradeoff_tikz(pareto):
    """Generate TikZ code for Energy-Delay Tradeoff"""
    if not pareto:
        print("⚠️  No pareto data for Energy-Delay plot")
        return "% No Pareto data available"
    
    energy = []
    delay = []
    
    for p in pareto:
        if not isinstance(p, dict):
            continue
        
        # Try different key variations
        e = p.get('Energy', p.get('energy', p.get('Energy_J', None)))
        d = p.get('Delay', p.get('delay', p.get('Delay_ms', None)))
        
        if e is not None and d is not None:
            energy.append(float(e))
            delay.append(float(d))
    
    if not energy:
        return "% No valid Energy-Delay points found"
    
    coords = " ".join([f"({e:.2f},{d:.3f})" for e, d in zip(energy, delay)])
    
    tikz = r"""\begin{tikzpicture}
\begin{axis}[
    width=0.8\textwidth,
    height=0.6\textwidth,
    xlabel={Energy (J)},
    ylabel={Delay (s)},
    grid=major,
    scatter/classes={a={mark=o,red}}
]
\addplot[scatter,only marks,mark size=2pt] coordinates {
""" + coords + r"""
};
\end{axis}
\end{tikzpicture}"""
    return tikz

def generate_comparison_table(cache):
    """Generate LaTeX comparison table with improvements"""
    
    # Calculate improvements
    u_hist = cache['U_history']
    d_hist = cache['Delta_history']
    o_hist = cache['Omega_history']
    e_hist = cache['Energy_history']
    delay_hist = cache['Delay_history']
    
    def improvement(hist, reverse=False):
        """Calculate percentage improvement (reverse=True for metrics where lower is better)"""
        if len(hist) < 2:
            return 0
        init = hist[0]
        final = hist[-1]
        if init == 0:
            return 0
        if reverse:
            return ((init - final) / init) * 100  # For error/delay: reduction is good
        return ((final - init) / init) * 100  # For utility/stability: increase is good
    
    u_imp = improvement(u_hist)
    d_imp = improvement(d_hist, reverse=True)
    o_imp = improvement(o_hist)
    e_imp = improvement(e_hist, reverse=True)
    delay_imp = improvement(delay_hist, reverse=True)
    
    table = r"""\begin{table}[h]
\centering
\caption{MATO-UAV Performance Summary}
\label{tab:mato_summary}
\begin{tabular}{lccr}
\toprule
\textbf{Metric} & \textbf{Initial} & \textbf{Final} & \textbf{Improvement} \\
\midrule
Utility ($U$)       & """ + f"{u_hist[0]:.3f}" + r""" & """ + f"{u_hist[-1]:.3f}" + r""" & """ + f"+{u_imp:.1f}\\%" + r""" \\
Error Rate ($\Delta$) & """ + f"{d_hist[0]:.4f}" + r""" & """ + f"{d_hist[-1]:.4f}" + r""" & """ + f"-{abs(d_imp):.1f}\\%" + r""" \\
Stability ($\Omega$) & """ + f"{o_hist[0]:.3f}" + r""" & """ + f"{o_hist[-1]:.3f}" + r""" & """ + f"+{o_imp:.1f}\\%" + r""" \\
Energy (J)          & """ + f"{e_hist[0]:.2f}" + r""" & """ + f"{e_hist[-1]:.2f}" + r""" & """ + f"-{abs(e_imp):.1f}\\%" + r""" \\
Delay (s)           & """ + f"{delay_hist[0]:.3f}" + r""" & """ + f"{delay_hist[-1]:.3f}" + r""" & """ + f"-{abs(delay_imp):.1f}\\%" + r""" \\
\bottomrule
\end{tabular}
\end{table}"""
    return table

def generate_master_usage():
    """Generate usage instructions"""
    usage = r"""% ============================================================
% MATO-UAV Chapter 5 Figures - Usage Guide
% ============================================================
% Generated TikZ codes for 8 figures + comparison table
%
% PREAMBLE REQUIREMENTS:
% \usepackage{pgfplots}
% \pgfplotsset{compat=1.18}
% \usepackage{booktabs}
%
% USAGE IN YOUR DOCUMENT:
%
% Figure 1: Utility Convergence
% \begin{figure}[h]
% \centering
% \input{tikz_codes/tikz_utility.tex}
% \caption{Utility convergence over training episodes}
% \label{fig:utility}
% \end{figure}
%
% Figure 2: Error Rate
% \begin{figure}[h]
% \centering
% \input{tikz_codes/tikz_error.tex}
% \caption{Error rate reduction during training}
% \label{fig:error}
% \end{figure}
%
% Figure 3: Stability
% \begin{figure}[h]
% \centering
% \input{tikz_codes/tikz_stability.tex}
% \caption{System stability improvement}
% \label{fig:stability}
% \end{figure}
%
% Figure 4: Pareto Front
% \begin{figure}[h]
% \centering
% \input{tikz_codes/tikz_pareto.tex}
% \caption{Pareto-optimal solutions (Utility vs Error)}
% \label{fig:pareto}
% \end{figure}
%
% Figure 5: Energy Consumption
% \begin{figure}[h]
% \centering
% \input{tikz_codes/tikz_energy.tex}
% \caption{Energy consumption over episodes}
% \label{fig:energy}
% \end{figure}
%
% Figure 6: Delay Performance
% \begin{figure}[h]
% \centering
% \input{tikz_codes/tikz_delay.tex}
% \caption{Delay performance improvement}
% \label{fig:delay}
% \end{figure}
%
% Figure 7: Energy-Delay Tradeoff
% \begin{figure}[h]
% \centering
% \input{tikz_codes/tikz_energy_delay_tradeoff.tex}
% \caption{Energy-Delay tradeoff analysis}
% \label{fig:energy_delay}
% \end{figure}
%
% Table: Comparison Table
% \input{tikz_codes/tikz_comparison_table.tex}
%
% ============================================================"""
    return usage

def main():
    print("="*60)
    print("Chapter 5 Report Generator - TikZ Version v7.4")
    print("="*60)
    
    # Load data
    cache, pareto = load_data()
    if cache is None:
        return
    
    print(f"\n📊 Generating TikZ codes...")
    
    # Generate all TikZ codes
    codes = {
        'tikz_utility.tex': generate_utility_tikz(cache),
        'tikz_error.tex': generate_error_tikz(cache),
        'tikz_stability.tex': generate_stability_tikz(cache),
        'tikz_energy.tex': generate_energy_tikz(cache),
        'tikz_delay.tex': generate_delay_tikz(cache),
        'tikz_pareto.tex': generate_pareto_tikz(pareto),
        'tikz_energy_delay_tradeoff.tex': generate_energy_delay_tradeoff_tikz(pareto),
        'tikz_comparison_table.tex': generate_comparison_table(cache),
        'master_usage.tex': generate_master_usage()
    }
    
    # Save all files
    for filename, content in codes.items():
        output_path = OUTPUT_DIR / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ {filename}")
    
    print(f"\n✅ All files saved to: {OUTPUT_DIR.absolute()}")
    print(f"\n📖 Read {OUTPUT_DIR / 'master_usage.tex'} for usage instructions!")
    print("="*60)

if __name__ == "__main__":
    main()
