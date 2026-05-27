"""
Two-Faces Print — MAAI Hackathon 2026
Hour 2.1: Relationship Analysis — Correlation, Mediation & Moderation
Author: Zahirah (CS2)
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# -- Load data (output of hour1.1) ---------------------------------------------

DATA_FILE = Path(__file__).parent.parent / 'data' / 'data_with_clusters.csv'
df = pd.read_csv(DATA_FILE)

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")

# -- Style ---------------------------------------------------------------------

CHARTS_DIR = Path(__file__).parent / 'charts'
CHARTS_DIR.mkdir(exist_ok=True)

NAVY  = '#1B2A4A'
BLUE  = '#2E6FD9'
CORAL = '#E8543A'
SLATE = '#F5F7FA'
GREY  = '#8FA3BF'

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.facecolor':     SLATE,
    'figure.facecolor':   'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   False,
    'axes.spines.bottom': False,
    'axes.grid':          True,
    'grid.color':         'white',
    'grid.linewidth':     1.2,
    'axes.labelcolor':    NAVY,
    'axes.titlecolor':    NAVY,
    'xtick.color':        GREY,
    'ytick.color':        GREY,
    'text.color':         NAVY,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}")


# -- Variables -----------------------------------------------------------------

iv         = 'Digital_Overload'
mediator   = 'Digital_Fatigue'
dvs        = ['Print_Trust', 'Print_Comprehension']
moderator  = 'Privacy_Concern'
rel_vars   = [iv, mediator, moderator] + dvs

# -- Correlation analysis ------------------------------------------------------

corr_matrix = df[rel_vars].corr()

ov_fat  = corr_matrix.loc[iv, mediator]
fat_pt  = corr_matrix.loc[mediator, 'Print_Trust']
ov_pc   = corr_matrix.loc[iv, 'Print_Comprehension']

print("\nKEY RELATIONSHIP INSIGHTS")
print(f"  Overload → Fatigue (a path):        r = {ov_fat:.2f}")
print(f"  Fatigue  → Print_Trust (b path):    r = {fat_pt:.2f}")
print(f"  Overload → Print_Comprehension:     r = {ov_pc:.2f}")

# -- Moderation check ----------------------------------------------------------

median_privacy = df[moderator].median()
high_privacy = df[df[moderator] > median_privacy]
low_privacy  = df[df[moderator] <= median_privacy]

r_high = high_privacy[iv].corr(high_privacy['Print_Trust'])
r_low  = low_privacy[iv].corr(low_privacy['Print_Trust'])

print(f"\nMODERATION ANALYSIS — effect of {moderator}")
print(f"  Overload → Print_Trust | High Privacy:  r = {r_high:.2f}")
print(f"  Overload → Print_Trust | Low Privacy:   r = {r_low:.2f}")
print(f"  Difference:                              Δr = {r_high - r_low:.2f}")

# -- Chart 1: Relationship correlation heatmap ---------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr_matrix, annot=True, fmt='.2f',
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    center=0, vmin=-1, vmax=1,
    linewidths=2, linecolor='white',
    annot_kws={'size': 12, 'weight': '500', 'color': NAVY},
    ax=ax, square=True, cbar_kws={'shrink': 0.7},
)
ax.set_title('Relationship map — IV → Mediator → DV',
             fontsize=13, fontweight='500', pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
save(fig, CHARTS_DIR / 'chart1_relationship_correlation.png')

# -- Chart 2: Moderation bar chart ---------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))
groups = ['High Privacy\nConcern', 'Low Privacy\nConcern']
values = [r_high, r_low]
colors = [CORAL if v > 0 else BLUE for v in values]

bars = ax.bar(groups, values, color=colors, width=0.45, zorder=3)
ax.axhline(0, color=GREY, linewidth=1.2, linestyle='--', zorder=2)
ax.set_ylabel('Correlation: Overload → Print_Trust', fontsize=11)
ax.set_title('Privacy Concern moderates the Overload → Trust path',
             fontsize=13, fontweight='500', pad=12)
ax.set_ylim(-0.2, 0.2)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + (0.008 if val >= 0 else -0.015),
            f'r = {val:.2f}', ha='center', va='bottom',
            fontsize=11, fontweight='500', color=NAVY)
save(fig, CHARTS_DIR / 'chart2_moderation_bar.png')

# -- Key findings --------------------------------------------------------------

print("\nKEY FINDINGS")
print(f"  Overload-Fatigue link:      r = {ov_fat:.2f}  (weak — consistent with hour2 regression)")
print(f"  Fatigue-Print_Trust link:   r = {fat_pt:.2f}  (minimal — mediation not supported)")
print(f"  Privacy moderation effect:  Δr = {r_high - r_low:.2f}  "
      f"({'visible' if abs(r_high - r_low) > 0.05 else 'minimal'} difference)")
print("\nHour 2.1 done — correlation charts ready")
