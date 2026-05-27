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

sns.set_context('talk')
sns.set_style('whitegrid')


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

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix, annot=True, fmt='.2f',
    cmap='Reds', linewidths=0.5, linecolor='white',
    ax=ax, square=True,
)
ax.set_title('Relationship Map: IV -> Mediator -> DV')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
save(fig, CHARTS_DIR / 'chart1_relationship_correlation.png')

# -- Chart 2: Moderation bar chart ---------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
groups = ['High Privacy\nConcern', 'Low Privacy\nConcern']
values = [r_high, r_low]

bars = ax.bar(groups, values, color=['#1f77b4', '#aec7e8'], width=0.45)
ax.axhline(0, color='grey', linewidth=1, linestyle='--')
ax.set_ylabel('Correlation: Overload -> Print_Trust')
ax.set_title('Moderation Effect of Privacy Concern')
ax.set_ylim(-0.2, 0.25)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + (0.008 if val >= 0 else -0.02),
            f'r = {val:.2f}', ha='center', va='bottom')
save(fig, CHARTS_DIR / 'chart2_moderation_bar.png')

# -- Key findings --------------------------------------------------------------

print("\nKEY FINDINGS")
print(f"  Overload-Fatigue link:      r = {ov_fat:.2f}  (weak — consistent with hour2 regression)")
print(f"  Fatigue-Print_Trust link:   r = {fat_pt:.2f}  (minimal — mediation not supported)")
print(f"  Privacy moderation effect:  Δr = {r_high - r_low:.2f}  "
      f"({'visible' if abs(r_high - r_low) > 0.05 else 'minimal'} difference)")
print("\nHour 2.1 done — correlation charts ready")
