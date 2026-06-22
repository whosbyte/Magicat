"""
Diabetes-Klassifikation -- Begleitende Python-Analyse zum Orange-Projekt.

Dieses Skript spiegelt exakt den Workflow, der in Orange aufgebaut wird, wider
(Datenreinigung -> EDA -> Decision Tree -> kNN -> 10 Train/Test-Splits ->
manuelle Gini-Verifikation). Es dient dazu, alle Zahlen und Abbildungen des
Berichts reproduzierbar und nachvollziehbar zu erzeugen. Die Algorithmen
(CART-Entscheidungsbaum mit Gini, kNN) entsprechen denen der Orange-Widgets
"Tree" und "kNN"; kleinere Abweichungen ergeben sich nur aus Zufalls-Seeds
und Default-Parametern (im Bericht dokumentiert).
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)

RNG = 42
FIG = "informatik-projekt-diabetes/figures"
DATA = "informatik-projekt-diabetes/data"
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# Spalten, in denen 0 medizinisch unmoeglich ist -> fehlende Werte
ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
results = {}

# ---------------------------------------------------------------- 1. Laden
raw = pd.read_csv(f"{DATA}/diabetes_raw.csv")
results['n_rows'] = len(raw)
results['class_counts'] = raw['Outcome'].value_counts().to_dict()

# ---------------------------------------------------------------- 2. Reinigung
missing = {}
clean = raw.copy()
for col in ZERO_AS_MISSING:
    n0 = int((clean[col] == 0).sum())
    missing[col] = {'zeros': n0, 'pct': round(100 * n0 / len(clean), 1)}
    clean[col] = clean[col].replace(0, np.nan)
results['missing_after_zero'] = missing

# Median-Imputation (robust gegen Ausreisser; pro Spalte)
medians = {}
for col in ZERO_AS_MISSING:
    m = clean[col].median()
    medians[col] = round(float(m), 2)
    clean[col] = clean[col].fillna(m)
results['imputation_medians'] = medians
clean.to_csv(f"{DATA}/diabetes_clean.csv", index=False)

X = clean[FEATURES].values
y = clean['Outcome'].values

# ---------------------------------------------------------------- 3. EDA
desc = clean[FEATURES + ['Outcome']].describe().round(2)
desc.to_csv(f"{DATA}/eda_describe.csv")
results['describe'] = json.loads(desc.to_json())

# 3a. Klassenverteilung
plt.figure(figsize=(4, 3.2))
vc = clean['Outcome'].map({0: 'kein Diabetes', 1: 'Diabetes'}).value_counts()
plt.bar(vc.index, vc.values, color=['#4C9F70', '#D1495B'])
for i, v in enumerate(vc.values):
    plt.text(i, v + 5, str(v), ha='center')
plt.title('Klassenverteilung (Zielvariable)')
plt.ylabel('Anzahl Personen')
plt.tight_layout(); plt.savefig(f"{FIG}/01_klassenverteilung.png", dpi=130); plt.close()

# 3b. Histogramme je Merkmal, getrennt nach Klasse
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, col in zip(axes.ravel(), FEATURES):
    ax.hist(clean[clean.Outcome == 0][col], bins=25, alpha=.6, label='kein Diabetes', color='#4C9F70')
    ax.hist(clean[clean.Outcome == 1][col], bins=25, alpha=.6, label='Diabetes', color='#D1495B')
    ax.set_title(col, fontsize=10)
axes[0, 0].legend(fontsize=8)
fig.suptitle('Verteilung der Merkmale nach Klasse', fontsize=13)
plt.tight_layout(); plt.savefig(f"{FIG}/02_histogramme.png", dpi=130); plt.close()

# 3c. Boxplots Glucose & BMI nach Klasse (die zwei staerksten Praediktoren)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, col in zip(axes, ['Glucose', 'BMI']):
    data = [clean[clean.Outcome == 0][col], clean[clean.Outcome == 1][col]]
    ax.boxplot(data, tick_labels=['kein\nDiabetes', 'Diabetes'])
    ax.set_title(col)
plt.tight_layout(); plt.savefig(f"{FIG}/03_boxplots.png", dpi=130); plt.close()

# 3d. Korrelationsmatrix
corr = clean[FEATURES + ['Outcome']].corr()
plt.figure(figsize=(7, 6))
im = plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046)
plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
plt.yticks(range(len(corr)), corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center',
                 fontsize=7, color='black')
plt.title('Korrelationsmatrix')
plt.tight_layout(); plt.savefig(f"{FIG}/04_korrelation.png", dpi=130); plt.close()
results['corr_with_outcome'] = corr['Outcome'].drop('Outcome').round(3).to_dict()


def metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
        'f1': round(f1_score(y_true, y_pred), 4),
        'auc': round(roc_auc_score(y_true, y_prob), 4),
    }


# ---------------------------------------------------------------- 4a. Interpretier-Baum (ganzer Datensatz)
# In Orange wird der Tree-Widget zur Visualisierung mit dem GESAMTEN Datensatz
# gespeist. Genau dieser Baum wird unten von Hand mit Gini verifiziert.
tree_full = DecisionTreeClassifier(criterion='gini', max_depth=4,
                                   min_samples_leaf=5, random_state=RNG)
tree_full.fit(X, y)
plt.figure(figsize=(20, 10))
plot_tree(tree_full, feature_names=FEATURES,
          class_names=['kein Diabetes', 'Diabetes'],
          filled=True, rounded=True, fontsize=9, impurity=True)
plt.title('Entscheidungsbaum (ganzer Datensatz, max_depth=4, min_samples_leaf=5)')
plt.tight_layout(); plt.savefig(f"{FIG}/05_entscheidungsbaum.png", dpi=130); plt.close()

root_feat_idx = tree_full.tree_.feature[0]
root_thr = tree_full.tree_.threshold[0]
results['root_split'] = {'feature': FEATURES[root_feat_idx],
                         'threshold': round(float(root_thr), 3)}
imp = dict(sorted(zip(FEATURES, tree_full.feature_importances_.round(4)),
                  key=lambda x: -x[1]))
results['tree_feature_importance'] = {k: float(v) for k, v in imp.items()}

# ---------------------------------------------------------------- 4b. Beispiel-Split (= Split 1 der 10er-Tabelle)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=1,
                                      stratify=y)
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5,
                              random_state=RNG).fit(Xtr, ytr)
tp = tree.predict(Xte); tpr = tree.predict_proba(Xte)[:, 1]
results['tree_main'] = metrics(yte, tp, tpr)
results['tree_main']['confusion'] = confusion_matrix(yte, tp).tolist()
disp = ConfusionMatrixDisplay(confusion_matrix(yte, tp),
                              display_labels=['kein Diabetes', 'Diabetes'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Konfusionsmatrix Entscheidungsbaum (Beispiel-Split, Testset)')
plt.tight_layout(); plt.savefig(f"{FIG}/06_konfusion_tree.png", dpi=130); plt.close()

# kNN: distanzbasiert -> Standardisierung (in Orange via 'Preprocess'/Normalize)
scaler = StandardScaler().fit(Xtr)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(scaler.transform(Xtr), ytr)
kp = knn.predict(scaler.transform(Xte)); kpr = knn.predict_proba(scaler.transform(Xte))[:, 1]
results['knn_main'] = metrics(yte, kp, kpr)
results['knn_main']['confusion'] = confusion_matrix(yte, kp).tolist()
disp = ConfusionMatrixDisplay(confusion_matrix(yte, kp),
                              display_labels=['kein Diabetes', 'Diabetes'])
disp.plot(cmap='Oranges', values_format='d')
plt.title('Konfusionsmatrix kNN (Beispiel-Split, Testset)')
plt.tight_layout(); plt.savefig(f"{FIG}/08_konfusion_knn.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 5. 10 Splits
rows = []
for i, seed in enumerate(range(1, 11), start=1):
    xa, xb, ya, yb = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    t = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5,
                               random_state=RNG).fit(xa, ya)
    sc = StandardScaler().fit(xa)
    k = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(xa), ya)
    mt = metrics(yb, t.predict(xb), t.predict_proba(xb)[:, 1])
    mk = metrics(yb, k.predict(sc.transform(xb)), k.predict_proba(sc.transform(xb))[:, 1])
    rows.append({'split': i, 'seed': seed,
                 **{f'tree_{m}': mt[m] for m in mt},
                 **{f'knn_{m}': mk[m] for m in mk}})
splits = pd.DataFrame(rows)
splits.to_csv(f"{DATA}/zehn_splits.csv", index=False)

summary = {}
for model in ['tree', 'knn']:
    summary[model] = {}
    for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        col = splits[f'{model}_{m}']
        summary[model][m] = {'mean': round(col.mean(), 4),
                             'min': round(col.min(), 4),
                             'max': round(col.max(), 4),
                             'std': round(col.std(), 4)}
results['ten_splits'] = splits.to_dict(orient='records')
results['ten_splits_summary'] = summary

# Abbildung: Accuracy ueber 10 Splits
plt.figure(figsize=(7, 4))
plt.plot(splits.split, splits.tree_accuracy, 'o-', label='Decision Tree', color='#1f77b4')
plt.plot(splits.split, splits.knn_accuracy, 's-', label='kNN (k=5)', color='#ff7f0e')
plt.axhline(summary['tree']['accuracy']['mean'], ls='--', color='#1f77b4', alpha=.5)
plt.axhline(summary['knn']['accuracy']['mean'], ls='--', color='#ff7f0e', alpha=.5)
plt.xticks(splits.split); plt.ylim(0.6, 0.85)
plt.xlabel('Train/Test-Aufteilung Nr.'); plt.ylabel('Accuracy')
plt.title('Accuracy ueber 10 zufaellige Aufteilungen')
plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG}/07_zehn_splits.png", dpi=130); plt.close()

# ---------------------------------------------------------------- 6. Gini von Hand
def gini(pos, neg):
    n = pos + neg
    if n == 0:
        return 0.0
    p = pos / n
    return 1 - p**2 - (1 - p)**2

def weighted_gini(df, col, thr):
    left = df[df[col] <= thr]; right = df[df[col] > thr]
    gl = gini((left.Outcome == 1).sum(), (left.Outcome == 0).sum())
    gr = gini((right.Outcome == 1).sum(), (right.Outcome == 0).sum())
    n = len(df)
    wg = len(left) / n * gl + len(right) / n * gr
    return {
        'threshold': thr,
        'n_total': n,
        'left': {'n': len(left), 'pos': int((left.Outcome == 1).sum()),
                 'neg': int((left.Outcome == 0).sum()), 'gini': round(gl, 4)},
        'right': {'n': len(right), 'pos': int((right.Outcome == 1).sum()),
                  'neg': int((right.Outcome == 0).sum()), 'gini': round(gr, 4)},
        'weighted_gini': round(wg, 4),
    }

# Wurzel-Gini des Gesamtdatensatzes
root_gini_full = gini((clean.Outcome == 1).sum(), (clean.Outcome == 0).sum())
results['gini'] = {'root_impurity_full': round(root_gini_full, 4)}
# (a) Vom Baum gewaehlte erste Entscheidung
results['gini']['model_choice'] = {
    'feature': FEATURES[root_feat_idx],
    **weighted_gini(clean, FEATURES[root_feat_idx], round(float(root_thr), 1))}
# (b) Frei gewaehlte Alternative: BMI <= 30 (klinische Adipositas-Grenze)
results['gini']['alternative'] = {
    'feature': 'BMI',
    **weighted_gini(clean, 'BMI', 30.0)}

# ---------------------------------------------------------------- speichern
with open(f"{DATA}/results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Konsolen-Zusammenfassung
print("=== ERGEBNISSE ===")
print("Zeilen:", results['n_rows'], "Klassen:", results['class_counts'])
print("\nFehlende Werte (0 -> NaN):")
for k, v in missing.items(): print(f"  {k}: {v['zeros']} ({v['pct']}%) -> Median {medians[k]}")
print("\nKorrelation mit Outcome:", results['corr_with_outcome'])
print("\nWurzel-Split des Baums:", results['root_split'])
print("Tree (70/30):", results['tree_main'])
print("kNN  (70/30):", results['knn_main'])
print("\n10-Splits Zusammenfassung:")
for model in summary:
    print(f"  {model}: acc {summary[model]['accuracy']}")
    print(f"        f1  {summary[model]['f1']}")
    print(f"        auc {summary[model]['auc']}")
print("\nGini Wurzel (gesamt):", results['gini']['root_impurity_full'])
print("Gini Modell-Wahl:", results['gini']['model_choice'])
print("Gini Alternative :", results['gini']['alternative'])
print("\nFeature-Wichtigkeit:", results['tree_feature_importance'])
print("\nFertig. Figuren in", FIG)
