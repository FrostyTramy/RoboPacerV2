"""
Grafic pentru un log de cruise_control.py - viteza reala vs tinta in timp,
excluzand esantioanele de "overshoot" (ex. varful de lansare) ca sa se vada
clar cat de constanta a fost viteza in restul cursei.

Rulare:
    python3 tools/plot_cruise_log.py                     # ultimul log din cruise_control/logs/
    python3 tools/plot_cruise_log.py cale/catre/log.csv
    python3 tools/plot_cruise_log.py --overshoot-factor 1.2 --out grafic.png
"""

import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")  # fara display - salveaza direct in fisier
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "cruise_control", "logs")


def find_latest_log():
    candidates = sorted(glob.glob(os.path.join(LOGS_DIR, "cruise_*.csv")))
    if not candidates:
        raise SystemExit(f"Niciun log gasit in {LOGS_DIR}")
    return candidates[-1]


def load_samples(csv_path):
    """Citeste elapsed_s/target_kmh/kmh/pulse_us dupa numele coloanelor (nu
    pozitie fixa) - compatibil si cu loguri vechi (fara effective_target_kmh)
    si cu cele noi. Se opreste la blocul SUMAR de la coada fisierului (nu e
    un rand CSV valid)."""
    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return samples
        idx = {name: i for i, name in enumerate(header)}
        required = ("elapsed_s", "target_kmh", "kmh", "pulse_us")
        if not all(name in idx for name in required):
            raise SystemExit(f"Coloane lipsa in {csv_path}: astept {required}, am gasit {header}")
        for row in reader:
            if len(row) != len(header):
                break  # blocul "--- SUMAR ---" / linie goala
            try:
                elapsed = float(row[idx["elapsed_s"]])
                target = float(row[idx["target_kmh"]])
                kmh = float(row[idx["kmh"]])
                pulse = float(row[idx["pulse_us"]])
            except ValueError:
                break
            samples.append((elapsed, target, kmh, pulse))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", help="Log cruise_control (implicit: cel mai recent)")
    ap.add_argument("--overshoot-factor", type=float, default=1.15,
                     help="Esantioanele cu kmh > tinta * factor sunt excluse din grafic (implicit 1.15 = 15%% peste tinta)")
    ap.add_argument("--out", default=None, help="Cale fisier PNG de iesire (implicit langa log)")
    args = ap.parse_args()

    csv_path = args.csv_path or find_latest_log()
    samples = load_samples(csv_path)
    if not samples:
        raise SystemExit(f"Niciun esantion valid in {csv_path}")

    kept = []
    excluded = []
    for elapsed, target, kmh, pulse in samples:
        ceiling = target * args.overshoot_factor if target > 0 else args.overshoot_factor
        if kmh > ceiling:
            excluded.append((elapsed, target, kmh, pulse))
        else:
            kept.append((elapsed, target, kmh, pulse))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    all_t = [s[0] for s in samples]
    all_target = [s[1] for s in samples]
    ax1.plot(all_t, all_target, "--", color="#888888", linewidth=1, label="tinta")
    ax1.plot([s[0] for s in kept], [s[2] for s in kept], color="#1f77b4", linewidth=1.2, label="viteza (pastrata)")
    if excluded:
        ax1.scatter([s[0] for s in excluded], [s[2] for s in excluded], color="#d62728", s=14,
                     zorder=5, label=f"exclus (>{args.overshoot_factor:.0%} din tinta)")
    ax1.set_ylabel("km/h")
    ax1.set_title(os.path.basename(csv_path))
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot([s[0] for s in samples], [s[3] for s in samples], color="#2ca02c", linewidth=1)
    ax2.set_ylabel("puls (us)")
    ax2.set_xlabel("timp (s)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    out_path = args.out or (os.path.splitext(csv_path)[0] + ".png")
    fig.savefig(out_path, dpi=130)
    print(f"Grafic salvat: {out_path}")
    print(f"Esantioane totale: {len(samples)} | pastrate: {len(kept)} | excluse: {len(excluded)}")


if __name__ == "__main__":
    main()
