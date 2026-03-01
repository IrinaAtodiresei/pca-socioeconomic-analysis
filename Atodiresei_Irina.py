import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# PROIECT DSAD - PCA pe indicatori World Bank (2022)
# Fișier input: worldbank_raw.csv (format WB: long)
# ============================================================

# -----------------------------
# 0) Config
# -----------------------------
INPUT_FILE = "worldbank_raw.csv"
YEAR_COL_RAW = "2022 [YR2022]"   # exact cum e în fișierul tău
COUNTRY_COL_RAW = "Country Name"
INDICATOR_COL_RAW = "Series Code"

variables = [
    "NY.GDP.PCAP.PP.CD",      # GDP per capita, PPP
    "SP.DYN.LE00.IN",         # Life expectancy
    "SP.DYN.IMRT.IN",         # Infant mortality
    "SE.SEC.ENRR",            # School enrollment, secondary
    "IT.NET.USER.ZS",         # Internet users
    "EG.ELC.ACCS.ZS",         # Access to electricity
    "SL.UEM.TOTL.ZS",         # Unemployment rate
    "SP.URB.TOTL.IN.ZS"       # Urban population
]

# -----------------------------
# 1) Citire date brute
# -----------------------------
df = pd.read_csv(INPUT_FILE)

# (Opțional) curățăm spațiile din numele coloanelor
df.columns = [c.strip() for c in df.columns]

# -----------------------------
# 2) Selectăm doar coloanele necesare și redenumim standard
# -----------------------------
df = df[[COUNTRY_COL_RAW, INDICATOR_COL_RAW, YEAR_COL_RAW]]
df.columns = ["Country Name", "Indicator Code", "2022"]

# -----------------------------
# 3) Convertim anul la numeric (rezolvă cazul în care e string)
# -----------------------------
df["2022"] = (
    df["2022"]
    .astype(str)
    .str.replace(",", ".", regex=False)  # dacă există zecimale cu virgulă
    .str.strip()
)
df["2022"] = pd.to_numeric(df["2022"], errors="coerce")

# -----------------------------
# 4) Păstrăm doar indicatorii proiectului
# -----------------------------
df = df[df["Indicator Code"].isin(variables)]

# -----------------------------
# 5) Pivot robust (long -> wide), agregare pentru duplicate
# -----------------------------
df_wide = df.pivot_table(
    index="Country Name",
    columns="Indicator Code",
    values="2022",
    aggfunc="mean"
)

# ne asigurăm că avem exact coloanele în ordinea dorită
df_wide = df_wide.reindex(columns=variables)

# -----------------------------
# 6) Curățare: eliminăm țări cu prea multe lipsuri + imputare mediană
# -----------------------------
df_wide = df_wide.dropna(thresh=6)  # minim 6 valori din 8
df_wide = df_wide.fillna(df_wide.median(numeric_only=True))

print("Număr de țări analizate:", df_wide.shape[0])
print("Număr de variabile:", df_wide.shape[1])

# -----------------------------
# 7) Standardizare
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_wide)

# -----------------------------
# 8) PCA complet (pentru scree)
# -----------------------------
pca = PCA()
pca.fit(X_scaled)

explained_variance = pca.explained_variance_ratio_
print("\nVarianță explicată (primele 5 componente):")
print(explained_variance[:5])

# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
plt.xlabel("Componentă principală")
plt.ylabel("Proporția varianței explicate")
plt.title("Scree Plot – PCA (World Bank, 2022)")
plt.show()

# -----------------------------
# 9) PCA cu 2 componente (pentru interpretare + plot)
# -----------------------------
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_scaled)

# scoruri (coordonatele țărilor în PCA)
scores = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=df_wide.index)

# încărcături factoriale (loadings)
loadings = pd.DataFrame(
    pca2.components_.T,
    columns=["PC1", "PC2"],
    index=df_wide.columns
)

print("\nÎncărcături factoriale (loadings):")
print(loadings)

# Scatter PC1 vs PC2
plt.figure(figsize=(8, 6))
plt.scatter(scores["PC1"], scores["PC2"], alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Țări în spațiul PCA (PC1 vs PC2)")
plt.show()

# -----------------------------
# 10) Salvare rezultate pentru lucrare (CSV)
# -----------------------------
loadings.to_csv("loadings_PC1_PC2.csv")
pd.DataFrame({"explained_variance_ratio": explained_variance}).to_csv("explained_variance.csv", index=False)
scores.to_csv("pca_scores_PC1_PC2.csv")

print("\nFișiere salvate:")
print("- loadings_PC1_PC2.csv")
print("- explained_variance.csv")
print("- pca_scores_PC1_PC2.csv")
