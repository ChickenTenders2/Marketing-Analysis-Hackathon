# Marketing Analysis Hackathon — Chart Guide

**Dataset:** 1,064 UK respondents | **Scale:** 1–5 Likert (1 = strongly disagree, 5 = strongly agree)

Use this guide as a reference when presenting. Each chart is described in plain language with the key talking point you should land.

---

## Hour 1 — Exploratory Data Analysis (EDA)
> *Who are our consumers and what do they feel?*

---

### Chart 1 — Correlation Matrix (`hour1/charts/chart1_correlation_heatmap.png`)

**What it shows:** A triangle heatmap showing how strongly every key variable relates to every other. Darker red = strong positive link. Darker blue = strong negative link. Numbers closer to 1 or −1 mean the relationship is stronger.

**Key talking point:**
- Privacy Concern and Print Trust have the strongest positive link — people who worry about their data online are more likely to trust print
- Online Intent is negatively linked to Print Trust — the more someone plans to read online, the less they trust print
- Digital Overload and Digital Fatigue are closely linked (as expected — they measure the same underlying feeling)

---

### Chart 2 — Print Trust by Demographics (`hour1/charts/chart2_trust_by_demographics.png`)

**What it shows:** Two side-by-side bar charts. Left: average Print Trust score for each age group (18–24 up to 65+). Right: average Print Trust split by gender. The dashed line shows the overall average across everyone.

**Key talking point:**
- Trust in print does not simply go up with age — the relationship is not linear, which challenges the assumption that "older = more loyal to print"
- The gender gap is very small — this is not a gender story, it is an age and attitude story

---

### Chart 3 — Digital Overload by Age (`hour1/charts/chart3_digital_overload_by_age.png`)

**What it shows:** Two charts side by side. Left: a stacked bar showing the spread of Digital Overload scores (1 to 5) within each age group — so you can see who scores high vs low. Right: a line chart comparing average Digital Overload and Privacy Concern scores across age groups.

**Key talking point:**
- Digital Overload is spread fairly evenly across age groups — it is not just a young person problem
- Privacy Concern stays consistently high across all ages, meaning every segment feels at risk online
- This sets up the argument that print is a relevant solution for a wide audience, not a niche older one

---

### Chart 4 — Variable Means Summary (`hour1/charts/chart4_variable_means.png`)

**What it shows:** A horizontal bar chart ranking all key variables by their average score. The dashed line at 3.0 is the neutral point. Bars to the right of it mean people tend to agree; bars near the left mean they tend to disagree.

**Key talking point:**
- Privacy Concern scores highest — it is the dominant feeling in the data
- Digital Fatigue also sits above neutral — people feel the effects of too much screen time
- Print Trust sits around the neutral mark (3.0) — it is not strongly loved or hated, which means there is an opportunity to shift it
- This chart is your scene-setter slide: "here is the emotional landscape of our consumer"

---

### Chart 5 — Cross-Tab Heatmap (`hour1/charts/chart5_crosstab_heatmap.png`)

**What it shows:** A grid where each cell shows the average Print Trust score for a specific combination of age group and gender. Darker blue = higher trust in that cell.

**Key talking point:**
- Lets you spot which demographic combination has the highest or lowest print trust at a glance
- Use this to identify your primary target segment — the cell with the highest trust score is where print investment is most likely to land

---

## Hour 2 — Predictive Modelling
> *What actually drives Print Trust, and by how much?*

---

### Chart 6 — Regression Coefficients (`hour2/charts/chart6_regression_coefficients.png`)

**What it shows:** A horizontal bar chart of the regression model results. Each bar shows how much a variable moves Print Trust up or down, holding everything else constant. The error bars show the uncertainty range. Highlighted (coral) bars are statistically significant — meaning the effect is real, not random noise.

**Key talking point:**
- Digital Overload and Privacy Concern alone are **not** significant predictors of Print Trust — this is the surprising finding most teams will miss
- The only significant predictor in the baseline model is the **55–64 age group**, who actually trust print *less* than 18–24 year olds — this directly challenges the "older = loyal print reader" assumption
- The low R² (~0.02) tells us no single variable drives trust on its own — the story is in the *combination*

---

### Chart 7 — Subgroup Print Trust (`hour2/charts/chart7_subgroup_print_trust.png`)

**What it shows:** Four bars comparing average Print Trust across four consumer groups: Low Overload + Low Privacy, High Overload only, High Privacy only, and High Overload + High Privacy combined. The dashed line is the overall average.

**Key talking point:**
- Print Trust is highest when a consumer has **both** high Digital Overload and high Privacy Concern (the coral bar)
- Having only one of those conditions does not produce the same effect — you need both together
- This combination represents around 19% of the sample (n≈206) and is your defined target segment for any print marketing investment

---

### Chart 8 — Mediation Pathway (`hour2/charts/chart8_mediation_pathway.png`)

**What it shows:** A diagram of the Baron & Kenny mediation test. It shows three boxes (Digital Overload → Digital Fatigue → Print Trust) with the statistical path values labelled on each arrow. Highlighted values are statistically significant.

**Key talking point:**
- The **a path** (Overload → Fatigue) is significant — feeling overloaded does drive digital fatigue
- The **b path** (Fatigue → Trust) is not significant — fatigue alone does not flip people to print
- This means Digital Fatigue is not the switching mechanism. The switch happens through attitude and privacy concern, which is a better story for a marketing campaign anyway

---

## Hour 1.1 — Clustering Analysis
> *Can we group consumers into meaningful segments?*

---

### Chart 1 — Elbow Method (`hour1.1/charts/chart1_elbow.png`)

**What it shows:** A line chart where the x-axis is the number of clusters (k) tested, and the y-axis is WCSS (Within-Cluster Sum of Squares) — basically a measure of how tightly packed the clusters are. The line bends sharply at the optimal k.

**Key talking point:**
- The bend in the curve occurs at **k = 3**, meaning 3 clusters captures the most useful structure without over-complicating it
- This justifies why we built 3 customer segments, not 2 or 5

---

### Chart 2 — PCA Cluster Map (`hour1.1/charts/chart2_pca_scatter.png`)

**What it shows:** A scatter plot where each dot is one respondent, plotted in 2D using PCA (Principal Component Analysis). PCA compresses all 9 survey variables down into 2 axes so we can see the clusters visually. Colour = which cluster that person belongs to.

**Key talking point:**
- The three clusters are visually separable — they are genuine groups, not just arbitrary splits
- The axes explain ~25% of total variance between them — enough to show clear separation
- Use this chart to show the audience that the segmentation is data-driven, not made up

---

### Chart 3 — Segment Persona Heatmap (`hour1.1/charts/chart3_cluster_heatmap.png`)

**What it shows:** A heatmap where each row is a cluster (0, 1, 2) and each column is a variable. Darker red = higher average score for that cluster on that variable.

**Key talking point:**
- **Cluster 0** — High Digital Overload (3.64) + moderate Print Trust (3.36): the digitally overwhelmed segment already leaning toward print
- **Cluster 1** — Low Overload (2.26) + Low Print Trust (2.55): disengaged from both digital stress and print — the hardest to convert
- **Cluster 2** — Low Overload (2.61) + moderate Print Trust (3.23): trusts print for other reasons, not because of digital fatigue — attitude-driven
- Cluster 0 is your primary target for a print campaign

---

## Hour 2.1 — Relationship Analysis
> *How do the variables connect, and does Privacy Concern change the picture?*

---

### Chart 1 — Relationship Correlation Map (`hour2.1/charts/chart1_relationship_correlation.png`)

**What it shows:** A square heatmap of correlations specifically between the IV (Digital Overload), mediator (Digital Fatigue), moderator (Privacy Concern), and both outcome variables (Print Trust and Print Comprehension). The diagonal is always 1 (a variable is perfectly correlated with itself).

**Key talking point:**
- The correlations between these variables are low across the board — reinforcing the hour 2 finding that no single variable drives print trust on its own
- This is not a weakness — it shows that consumer behaviour is complex, and a nuanced segmentation approach (like the clusters) is the right strategy

---

### Chart 2 — Moderation Bar Chart (`hour2.1/charts/chart2_moderation_bar.png`)

**What it shows:** Two bars comparing the correlation between Digital Overload and Print Trust for two groups: people with high Privacy Concern versus people with low Privacy Concern.

**Key talking point:**
- For people with **high Privacy Concern**, the link between Digital Overload and Print Trust is stronger (r = 0.17) — Privacy Concern amplifies the effect
- For people with **low Privacy Concern**, the link is near zero (r = 0.02) — Digital Overload alone does very little
- This confirms Privacy Concern as a **moderator** — it switches the overload-to-trust pathway on or off
- Practical implication: a print campaign that leans into privacy and data security messaging will be most effective for the high-concern segment

---

## Project Structure

```
hour1/          EDA — who the consumers are and what they feel
hour2/          Modelling — what drives print trust and by how much
hour1.1/        Clustering — three data-driven consumer segments
hour2.1/        Relationships — how variables connect and interact
data/           Shared dataset (Excel source + CSV exports)
requirements.txt  Python dependencies
```
