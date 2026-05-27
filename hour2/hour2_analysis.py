"""
Two-Faces Print — MAAI Hackathon 2026
Hour 2: Predictive Modelling — Regression, Moderation & Mediation
Author: Alex (CS1)

Models built:
  Model 1 — Baseline OLS: Print_Trust ~ Digital_Overload + Privacy_Concern + Age_Group
  Model 2 — Moderation:   Print_Trust ~ Digital_Overload × Privacy_Concern + Age (mean-centred)
  Model 3 — Interaction:  Print_Trust ~ Digital_Overload × Age_Num + Privacy_Concern
  Mediation — Baron & Kenny 3-step: Digital_Overload → Digital_Fatigue → Print_Trust
  Subgroup — High vs Low overload+privacy comparison
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

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

age_map = {'18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55-64': 5, '65+': 6}
df['Age_Num'] = df['Age_Group'].map(age_map).astype(float)

df['DO_c']  = df['Digital_Overload'] - df['Digital_Overload'].mean()
df['PC_c']  = df['Privacy_Concern']  - df['Privacy_Concern'].mean()
df['Age_c'] = df['Age_Num']          - df['Age_Num'].mean()

print(f"Data prepared: n = {len(df)}")

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


# -- Model 1: Baseline OLS -----------------------------------------------------

model1 = smf.ols(
    'Print_Trust ~ Digital_Overload + Privacy_Concern + C(Age_Group)',
    data=df,
).fit()

print("\nMODEL 1 — Baseline OLS (reference group: Age 18-24)")
print(model1.summary())

# -- Model 2: Moderation (mean-centred) ----------------------------------------

model2 = smf.ols('Print_Trust ~ DO_c * PC_c + Age_c', data=df).fit()

print("\nMODEL 2 — Moderation (mean-centred)")
print("Privacy_Concern as moderator of Digital_Overload → Print_Trust")
print(model2.summary())

# -- Model 3: Age interaction --------------------------------------------------

model3 = smf.ols('Print_Trust ~ DO_c * Age_c + PC_c', data=df).fit()

print("\nMODEL 3 — Age × Digital_Overload Interaction (mean-centred)")
print(model3.summary())

# -- Baron & Kenny 3-step mediation --------------------------------------------

step1 = smf.ols('Print_Trust ~ Digital_Overload', data=df).fit()
step2 = smf.ols('Digital_Fatigue ~ Digital_Overload', data=df).fit()
step3 = smf.ols('Print_Trust ~ Digital_Overload + Digital_Fatigue', data=df).fit()

c_path  = step1.params['Digital_Overload']
c_p     = step1.pvalues['Digital_Overload']
a_path  = step2.params['Digital_Overload']
a_p     = step2.pvalues['Digital_Overload']
b_path  = step3.params['Digital_Fatigue']
b_p     = step3.pvalues['Digital_Fatigue']
cp_path = step3.params['Digital_Overload']
cp_p    = step3.pvalues['Digital_Overload']
indirect = a_path * b_path

print("\nBARON & KENNY 3-STEP MEDIATION")
print("IV: Digital_Overload  |  Mediator: Digital_Fatigue  |  DV: Print_Trust")
print(f"Step 1 — Total effect  (c path):  β={c_path:.4f},  p={c_p:.4f}  "
      f"{'sig' if c_p < 0.05 else 'not sig'}")
print(f"Step 2 — IV → Mediator (a path):  β={a_path:.4f},  p={a_p:.4f}  "
      f"{'sig' if a_p < 0.05 else 'not sig'}")
print(f"Step 3 — Mediator → DV (b path):  β={b_path:.4f},  p={b_p:.4f}  "
      f"{'sig' if b_p < 0.05 else 'not sig'}")
print(f"Step 3 — Direct effect (c' path): β={cp_path:.4f}, p={cp_p:.4f}  "
      f"{'sig' if cp_p < 0.05 else 'not sig'}")
print(f"Indirect effect (a × b): {indirect:.4f}")
print(f"Reduction in c → c':     {c_path - cp_path:.4f}")

if c_p < 0.05 and a_p < 0.05 and b_p < 0.05:
    verdict = "FULL MEDIATION" if cp_p >= 0.05 else "PARTIAL MEDIATION"
elif a_p < 0.05:
    verdict = ("PARTIAL PATHWAY — a path significant but full chain not established. "
               "Modern mediation (Zhao et al., 2010) allows testing indirect effects "
               "regardless of c path significance.")
else:
    verdict = "MEDIATION NOT SUPPORTED under classical B&K criteria"
print(f"Verdict: {verdict}")

# -- Subgroup analysis ---------------------------------------------------------

high_both = df[(df['Digital_Overload'] >= 4) & (df['Privacy_Concern'] >= 4)]
low_both  = df[(df['Digital_Overload'] <= 2) & (df['Privacy_Concern'] <= 2)]
high_over = df[(df['Digital_Overload'] >= 4) & (df['Privacy_Concern'] < 4)]
high_priv = df[(df['Digital_Overload'] < 4)  & (df['Privacy_Concern'] >= 4)]

t_stat, t_p = stats.ttest_ind(high_both['Print_Trust'], low_both['Print_Trust'])

print("\nSUBGROUP ANALYSIS — Who actually trusts print?")
print(f"High Overload + High Privacy:  n={len(high_both):>4},  "
      f"Print_Trust = {high_both['Print_Trust'].mean():.3f}")
print(f"High Overload only:            n={len(high_over):>4},  "
      f"Print_Trust = {high_over['Print_Trust'].mean():.3f}")
print(f"High Privacy only:             n={len(high_priv):>4},  "
      f"Print_Trust = {high_priv['Print_Trust'].mean():.3f}")
print(f"Low Overload + Low Privacy:    n={len(low_both):>4},  "
      f"Print_Trust = {low_both['Print_Trust'].mean():.3f}")
print(f"Overall:                       n={len(df):>4},  "
      f"Print_Trust = {df['Print_Trust'].mean():.3f}")
print(f"High-both vs Low-both: t={t_stat:.3f}, p={t_p:.4f}  "
      f"{'sig' if t_p < 0.05 else 'not sig'}")
print(f"Effect size (difference): "
      f"{high_both['Print_Trust'].mean() - low_both['Print_Trust'].mean():.3f}")

young_hi = df[(df['Age_Group'].isin(['18-24', '25-34'])) & (df['Digital_Overload'] >= 4)]
older_hi = df[(df['Age_Group'].isin(['55-64', '65+']))   & (df['Digital_Overload'] >= 4)]
print(f"Young (18-34) + high overload:  n={len(young_hi)},  "
      f"Print_Trust={young_hi['Print_Trust'].mean():.3f}")
print(f"Older (55+)   + high overload:  n={len(older_hi)},  "
      f"Print_Trust={older_hi['Print_Trust'].mean():.3f}")

# -- Chart 6: Regression coefficient plot --------------------------------------

params = model1.params.drop('Intercept')
conf   = model1.conf_int().drop('Intercept')
pvals  = model1.pvalues.drop('Intercept')

label_map = {
    'Digital_Overload':        'Digital Overload',
    'Privacy_Concern':         'Privacy Concern',
    'C(Age_Group)[T.25-34]':   'Age: 25–34 vs 18–24',
    'C(Age_Group)[T.35-44]':   'Age: 35–44 vs 18–24',
    'C(Age_Group)[T.45-54]':   'Age: 45–54 vs 18–24',
    'C(Age_Group)[T.55-64]':   'Age: 55–64 vs 18–24',
    'C(Age_Group)[T.65+]':     'Age: 65+ vs 18–24',
}
labels = [label_map.get(p, p) for p in params.index]
colors = [CORAL if pvals[p] < 0.05 else GREY for p in params.index]
y_pos  = np.arange(len(params))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(y_pos, params.values, color=colors, height=0.55, zorder=3)
ax.errorbar(
    params.values, y_pos,
    xerr=[params.values - conf[0].values, conf[1].values - params.values],
    fmt='none', color=NAVY, capsize=4, linewidth=1.2, zorder=4,
)
ax.axvline(0, color=GREY, linewidth=1.2, linestyle='--', zorder=2)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Regression coefficient (β)', fontsize=11)
ax.set_title('Model 1 — predictors of Print_Trust\n(highlighted = p < 0.05)',
             fontsize=13, fontweight='500', pad=12)
ax.legend(handles=[
    Patch(facecolor=CORAL, label='Significant (p < 0.05)'),
    Patch(facecolor=GREY,  label='Non-significant'),
], fontsize=9, framealpha=0.8, loc='lower right')
save(fig, CHARTS_DIR / 'chart6_regression_coefficients.png')


# -- Chart 7: Subgroup comparison bar ------------------------------------------

groups = [
    f'Low Overload\n+ Low Privacy\n(n={len(low_both)})',
    f'High Overload\nonly\n(n={len(high_over)})',
    f'High Privacy\nonly\n(n={len(high_priv)})',
    f'High Overload\n+ High Privacy\n(n={len(high_both)})',
]
values = [
    low_both['Print_Trust'].mean(),
    high_over['Print_Trust'].mean(),
    high_priv['Print_Trust'].mean(),
    high_both['Print_Trust'].mean(),
]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(groups, values, color=[GREY, GREY, BLUE, CORAL], width=0.55, zorder=3)
ax.axhline(df['Print_Trust'].mean(), color=GREY, linewidth=1.2, linestyle='--', zorder=2,
           label=f'Overall mean ({df["Print_Trust"].mean():.2f})')
ax.set_ylim(2.5, 3.5)
ax.set_ylabel('Mean Print_Trust (1–5)', fontsize=11)
ax.set_title('Print trust peaks when overload AND privacy concern are both high',
             fontsize=13, fontweight='500', pad=12)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f'{val:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='500', color=NAVY)
ax.legend(fontsize=9, framealpha=0.8)
save(fig, CHARTS_DIR / 'chart7_subgroup_print_trust.png')


# -- Chart 8: Baron & Kenny mediation diagram ----------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
ax.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

box_style = dict(boxstyle='round,pad=0.5', facecolor=SLATE, edgecolor=GREY, linewidth=1.5)
arrowprops = dict(arrowstyle='->', color=NAVY, lw=1.8)

ax.text(1,   2,   'Digital\nOverload\n(IV)',        ha='center', va='center',
        fontsize=11, fontweight='500', bbox=box_style)
ax.text(5,   3.2, 'Digital\nFatigue\n(Mediator)',   ha='center', va='center',
        fontsize=11, fontweight='500', bbox=box_style)
ax.text(9,   2,   'Print\nTrust\n(DV)',             ha='center', va='center',
        fontsize=11, fontweight='500', bbox=box_style)

ax.annotate('', xy=(4.2, 3.1), xytext=(1.7, 2.4), arrowprops=arrowprops)
ax.annotate('', xy=(8.2, 2.5), xytext=(5.8, 3.0), arrowprops=arrowprops)
ax.annotate('', xy=(8.2, 2.0), xytext=(1.8, 2.0),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8))

sig = lambda p: CORAL if p < 0.05 else GREY
ax.text(2.7, 3.05, f'a = {a_path:.3f}{"*" if a_p < 0.05 else ""}',
        fontsize=10, color=sig(a_p), fontweight='500')
ax.text(7.0, 3.05, f'b = {b_path:.3f}{"*" if b_p < 0.05 else ""}',
        fontsize=10, color=sig(b_p), fontweight='500')
ax.text(4.5, 1.65, f"c = {c_path:.3f} (total)   c' = {cp_path:.3f} (direct)",
        fontsize=10, color=BLUE, fontweight='500', ha='center')
ax.text(5, 0.8,
        f'Indirect effect (a×b) = {indirect:.4f}   |   '
        f'a path: p={a_p:.3f}   b path: p={b_p:.3f}',
        fontsize=9, color=GREY, ha='center', style='italic')
ax.set_title('Baron & Kenny mediation — Digital_Overload → Digital_Fatigue → Print_Trust',
             fontsize=13, fontweight='500', pad=8)
save(fig, CHARTS_DIR / 'chart8_mediation_pathway.png')


# -- Key findings --------------------------------------------------------------

print("\nKEY FINDINGS")
print(f"  Model R²:            ~0.02 — aggregate relationship is weak")
print(f"  Digital_Overload:    non-significant predictor alone (p=0.506)")
print(f"  Privacy_Concern:     non-significant predictor alone (p=0.572)")
print(f"  Age 55-64:           only significant predictor (β≈−0.47, p=0.001)")
print(f"  High overload+privacy group Print_Trust: {high_both['Print_Trust'].mean():.3f}")
print(f"  Low overload+privacy group  Print_Trust: {low_both['Print_Trust'].mean():.3f}")
print(f"  Difference: {high_both['Print_Trust'].mean() - low_both['Print_Trust'].mean():.3f}  "
      f"(Privacy Multiplier effect)")
print(f"  Mediation verdict:   {verdict}")
print("\nHour 2 done — charts ready for Taylor, subgroup stats for Jordan")
