"""
Two-Faces Print — MAAI Hackathon 2026
Hour 1.1: Clustering Analysis — K-Means Segmentation
Author: Zahirah (CS2)
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# -- Load data -----------------------------------------------------------------

DATA_FILE = Path(__file__).parent.parent / 'data' / 'data.csv'
df_raw = pd.read_csv(DATA_FILE)

column_mapping = {
    'I get a better understanding of the story when reading news in print rather than online':
        'Print_Comprehension',
    'I trust news stories more when they are in printed newspapers, rather than when they are online':
        'Print_Trust',
    'I spend too much time on electronic devices':
        'Digital_Fatigue',
    'I suffer from a digital overload':
        'Digital_Overload',
    'I am increasingly concerned that my personal information held electronically is at risk of being hacked, stolen, lost or damaged':
        'Privacy_Concern',
    'In the future, I intend to read more news online':
        'Future_Online_Intent',
    'I would be concerned if printed newspapers were to disappear':
        'Print_Disappearance_Concern',
    'I am concerned that the overuse of electronic devices could be damaging to my health (eyestrain, sleep deprivation, headaches)':
        'Health_Concern',
    'I think children and students learn more when reading printed, rather than digital, books and course materials':
        'Student_Print_Preference',
}

df = df_raw.copy()
for col in df.columns:
    for search_text, new_name in column_mapping.items():
        if search_text.lower() in col.lower():
            df.rename(columns={col: new_name}, inplace=True)

target_vars = [
    'Print_Comprehension', 'Print_Trust',
    'Digital_Fatigue', 'Digital_Overload', 'Privacy_Concern',
]
all_vars = target_vars + [
    'Future_Online_Intent', 'Print_Disappearance_Concern',
    'Health_Concern', 'Student_Print_Preference',
]

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


# -- Scale & cluster -----------------------------------------------------------

X = df[all_vars].dropna().copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -- Chart 1: Elbow plot -------------------------------------------------------

inertias = []
k_range = range(2, 9)
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(k_range), inertias, marker='o', linewidth=2.2,
        color=BLUE, markersize=8, zorder=4)
ax.fill_between(list(k_range), inertias, alpha=0.1, color=BLUE)
ax.axvline(3, color=CORAL, linewidth=1.5, linestyle='--', zorder=3,
           label='Chosen k = 3')
ax.set_xlabel('Number of clusters (k)', fontsize=11)
ax.set_ylabel('Inertia (within-cluster sum of squares)', fontsize=11)
ax.set_title('Elbow plot — choosing optimal k for K-Means',
             fontsize=13, fontweight='500', pad=12)
ax.legend(fontsize=9, framealpha=0.8)
save(fig, CHARTS_DIR / 'chart1_elbow.png')

# -- Fit final model (k=3) -----------------------------------------------------

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_profiles = df.groupby('Cluster')[all_vars].mean()
cluster_profiles['Size'] = df['Cluster'].value_counts().sort_index()

cluster_profiles.to_csv(Path(__file__).parent / 'cluster_profiles.csv')
df.to_csv(Path(__file__).parent.parent / 'data' / 'data_with_clusters.csv', index=False)

print(f"\nCluster sizes:\n{df['Cluster'].value_counts().sort_index()}")
print(f"\nCluster profiles:\n{cluster_profiles.round(2)}")

# -- Chart 2: Cluster heatmap --------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    cluster_profiles[all_vars], annot=True, fmt='.2f',
    cmap=sns.light_palette(CORAL, as_cmap=True),
    linewidths=2, linecolor='white',
    annot_kws={'size': 10, 'weight': '500', 'color': NAVY},
    ax=ax, cbar_kws={'shrink': 0.6},
)
ax.set_title('Segment persona heatmap — mean scores by cluster',
             fontsize=13, fontweight='500', pad=12)
ax.set_xlabel('Variable', fontsize=11)
ax.set_ylabel('Cluster', fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
save(fig, CHARTS_DIR / 'chart2_cluster_heatmap.png')

# -- Key findings --------------------------------------------------------------

print("\nKEY FINDINGS")
for i, row in cluster_profiles.iterrows():
    print(f"  Cluster {i} (n={int(row['Size'])}): "
          f"Print_Trust={row['Print_Trust']:.2f}  "
          f"Digital_Overload={row['Digital_Overload']:.2f}  "
          f"Privacy_Concern={row['Privacy_Concern']:.2f}")
print("\nHour 1.1 done — data_with_clusters.csv ready for hour2.1")
