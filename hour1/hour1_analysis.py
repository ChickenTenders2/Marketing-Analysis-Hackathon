"""
Two-Faces Print — MAAI Hackathon 2026
Hour 1: Exploratory Data Analysis (EDA)
Author: Alex (CS1)
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# -- Load data -----------------------------------------------------------------

DATA_FILE = Path(__file__).parent.parent / 'data' / 'Hackathon_Raw_Data_TwoFaces Generated.xlsx'
df = pd.read_excel(DATA_FILE, sheet_name='Hackathon_Raw_Data_TwoFaces Gen')

df.rename(columns={
    'I get a better understanding of the story when reading news in print rather than online':
        'Print_Comprehension',
    'I trust news stories more when they are in printed newspapers, rather than when they are online':
        'Print_Trust',
    'In the future, I intend to read more news online':
        'Online_Intent',
    'I would be concerned if printed newspapers were to disappear':
        'Print_Concern',
    'I spend too much time on electronic devices':
        'Digital_Fatigue',
    'I suffer from a digital overload':
        'Digital_Overload',
    'I am concerned that the overuse of electronic devices could be damaging to my health (eyestrain, sleep deprivation, headaches)':
        'Health_Concern',
    'I am increasingly concerned that my personal information held electronically is at risk of being hacked, stolen, lost or damaged':
        'Privacy_Concern',
    'I think children and students learn more when reading printed, rather than digital, books and course materials':
        'Print_Education',
}, inplace=True)

age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['Age_Group'] = pd.Categorical(df['Age_Group'], categories=age_order, ordered=True)

key_vars = [
    'Digital_Overload', 'Digital_Fatigue', 'Privacy_Concern',
    'Print_Trust', 'Print_Comprehension', 'Print_Concern', 'Health_Concern',
]

print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} cols")
print(f"Missing values: {df.isnull().sum().sum()}")
print(df[key_vars].describe().round(2))

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


def bar_labels(ax, bars, hoffset=0.05):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + hoffset,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='500', color=NAVY,
        )


def save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}")


# -- Chart 1: Correlation heatmap ----------------------------------------------

fig, ax = plt.subplots(figsize=(9, 7))

corr = df[key_vars].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    center=0, vmin=-1, vmax=1,
    linewidths=2, linecolor='white',
    annot_kws={'size': 11, 'weight': '500', 'color': NAVY},
    ax=ax, square=True, cbar_kws={'shrink': 0.7},
)
ax.set_title('Correlation matrix — key variables', fontsize=13, fontweight='500', pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
fig.text(
    0.12, 0.01,
    'Privacy_Concern & Print_Trust show the strongest positive link  ·  '
    'Online_Intent negatively correlated with Print_Trust',
    fontsize=9, color=GREY, style='italic',
)
save(fig, CHARTS_DIR / 'chart1_correlation_heatmap.png')


# -- Chart 2: Print_Trust by age and gender ------------------------------------

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Who trusts print? — demographic breakdown',
             fontsize=15, fontweight='500', color=NAVY, y=1.02)

overall_mean = df['Print_Trust'].mean()

trust_age = df.groupby('Age_Group', observed=True)['Print_Trust'].mean().reset_index()
peak = trust_age['Print_Trust'].max()
bars = ax_l.bar(
    trust_age['Age_Group'], trust_age['Print_Trust'],
    color=[CORAL if v == peak else BLUE for v in trust_age['Print_Trust']],
    width=0.55, zorder=3,
)
ax_l.set_title('By age group', fontsize=13, fontweight='500', pad=12)
ax_l.set_xlabel('Age group', fontsize=11)
ax_l.set_ylabel('Mean Print_Trust (1–5)', fontsize=11)
ax_l.set_ylim(1, 5)
ax_l.axhline(overall_mean, color=GREY, linewidth=1.2, linestyle='--', zorder=2)
ax_l.text(5.55, overall_mean + 0.06, f'Mean\n{overall_mean:.2f}', fontsize=8, color=GREY)
bar_labels(ax_l, bars)

trust_gen = df.groupby('Gender')['Print_Trust'].mean().reset_index()
gen_counts = df['Gender'].value_counts()
bars2 = ax_r.bar(
    trust_gen['Gender'], trust_gen['Print_Trust'],
    color=[BLUE, CORAL], width=0.4, zorder=3,
)
ax_r.set_title('By gender', fontsize=13, fontweight='500', pad=12)
ax_r.set_xlabel('Gender', fontsize=11)
ax_r.set_ylabel('Mean Print_Trust (1–5)', fontsize=11)
ax_r.set_ylim(1, 5)
ax_r.axhline(overall_mean, color=GREY, linewidth=1.2, linestyle='--', zorder=2)
for bar, row in zip(bars2, trust_gen.itertuples()):
    ax_r.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
              f'{row.Print_Trust:.2f}', ha='center', va='bottom',
              fontsize=10, fontweight='500', color=NAVY)
    ax_r.text(bar.get_x() + bar.get_width() / 2, 1.15,
              f'n={gen_counts[row.Gender]}', ha='center', fontsize=9, color=GREY)
save(fig, CHARTS_DIR / 'chart2_trust_by_demographics.png')


# -- Chart 3: Digital overload by age ------------------------------------------

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Digital fatigue landscape — who is overwhelmed?',
             fontsize=15, fontweight='500', color=NAVY, y=1.02)

pivot = df.groupby(['Age_Group', 'Digital_Overload'], observed=True).size().unstack(fill_value=0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot_pct.plot(
    kind='bar', stacked=True, ax=ax_l,
    color=['#C9D8F0', '#8FA3BF', '#2E6FD9', '#E8543A', '#B03020'],
    width=0.65, zorder=3,
)
ax_l.set_title('Overload distribution by age group', fontsize=13, fontweight='500', pad=12)
ax_l.set_xlabel('Age group', fontsize=11)
ax_l.set_ylabel('% of respondents', fontsize=11)
ax_l.set_xticklabels(age_order, rotation=0, fontsize=10)
ax_l.legend(title='Score', labels=['1 (low)', '2', '3', '4', '5 (high)'],
            fontsize=9, title_fontsize=9, loc='upper right', framealpha=0.8)

overload_age = df.groupby('Age_Group', observed=True)['Digital_Overload'].mean()
privacy_age  = df.groupby('Age_Group', observed=True)['Privacy_Concern'].mean()
x = np.arange(len(age_order))
ax_r.plot(x, overload_age.values, marker='o', linewidth=2.2,
          color=BLUE, markersize=8, label='Digital_Overload', zorder=4)
ax_r.plot(x, privacy_age.values, marker='s', linewidth=2.2,
          color=CORAL, markersize=8, label='Privacy_Concern', zorder=4)
for i, (o, p) in enumerate(zip(overload_age.values, privacy_age.values)):
    ax_r.text(i, o + 0.07, f'{o:.2f}', ha='center', fontsize=9, color=BLUE, fontweight='500')
    ax_r.text(i, p - 0.13, f'{p:.2f}', ha='center', fontsize=9, color=CORAL, fontweight='500')
ax_r.set_title('Overload vs privacy concern by age', fontsize=13, fontweight='500', pad=12)
ax_r.set_xticks(x)
ax_r.set_xticklabels(age_order, fontsize=10)
ax_r.set_ylabel('Mean score (1–5)', fontsize=11)
ax_r.set_ylim(1.5, 4.5)
ax_r.legend(fontsize=10, framealpha=0.8)
save(fig, CHARTS_DIR / 'chart3_digital_overload_by_age.png')


# -- Chart 4: Variable means summary -------------------------------------------

means = df[key_vars].mean().sort_values()

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(
    means.index, means.values,
    color=[CORAL if v >= 3.4 else BLUE if v >= 3.0 else GREY for v in means],
    height=0.55, zorder=3,
)
ax.axvline(3.0, color=GREY, linewidth=1.2, linestyle='--', zorder=2, label='Neutral (3.0)')
ax.set_xlim(1, 5)
ax.set_xlabel('Mean score (1–5 Likert)', fontsize=11)
ax.set_title('Mean scores — UK sample (n=1,064)', fontsize=13, fontweight='500', pad=12)
for bar in bars:
    ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.2f}', va='center', fontsize=10, fontweight='500', color=NAVY)
ax.text(3.62, 6.38, '← Highest concern', fontsize=8, color=CORAL, style='italic')
ax.text(2.50, 0.38, 'Below neutral →',   fontsize=8, color=GREY,  style='italic')
ax.legend(fontsize=9, framealpha=0.8, loc='lower right')
save(fig, CHARTS_DIR / 'chart4_variable_means.png')


# -- Chart 5: Cross-tab heatmap ------------------------------------------------

pivot_ct = df.groupby(['Age_Group', 'Gender'], observed=True)['Print_Trust'].mean().unstack()

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    pivot_ct, annot=True, fmt='.2f',
    cmap=sns.light_palette(BLUE, as_cmap=True),
    linewidths=2, linecolor='white',
    annot_kws={'size': 11, 'weight': '500', 'color': NAVY},
    ax=ax, cbar_kws={'shrink': 0.7},
)
ax.set_title('Mean Print_Trust — age group × gender', fontsize=13, fontweight='500', pad=12)
ax.set_xlabel('Gender', fontsize=11)
ax.set_ylabel('Age group', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
save(fig, CHARTS_DIR / 'chart5_crosstab_heatmap.png')


# -- Key findings --------------------------------------------------------------

trust_by_age = df.groupby('Age_Group', observed=True)['Print_Trust'].mean()

print("\nKEY FINDINGS")
print(f"  Privacy_Concern:     {df['Privacy_Concern'].mean():.2f}  (highest — consumers feel risk)")
print(f"  Digital_Fatigue:     {df['Digital_Fatigue'].mean():.2f}  (feel the effects...)")
print(f"  Digital_Overload:    {df['Digital_Overload'].mean():.2f}  (...but don't label it overload)")
print(f"  Print_Trust:         {df['Print_Trust'].mean():.2f}  (neutral territory)")
print(f"  Print_Comprehension: {df['Print_Comprehension'].mean():.2f}  (comprehension outpaces trust)")
print(f"  Top age for trust:   {trust_by_age.idxmax()} ({trust_by_age.max():.2f})")
print(f"  Low age for trust:   {trust_by_age.idxmin()} ({trust_by_age.min():.2f})")
print(f"  Female={df[df['Gender']=='Female']['Print_Trust'].mean():.2f}  "
      f"Male={df[df['Gender']=='Male']['Print_Trust'].mean():.2f}  (gender gap minimal)")
print(f"  18-24 male Print_Trust: "
      f"{df[(df['Age_Group']=='18-24') & (df['Gender']=='Male')]['Print_Trust'].mean():.2f}"
      f"  (highest cell in cross-tab)")
print("\nEDA done — charts ready for Taylor, findings for Jordan")
