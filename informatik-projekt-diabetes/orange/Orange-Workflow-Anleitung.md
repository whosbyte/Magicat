# Orange-Workflow – Schritt-für-Schritt-Anleitung

Diese Anleitung baut den kompletten Analyse-Workflow in **Orange Data Mining**
nach. Sie deckt alle in der Aufgabenstellung geforderten Schritte ab. Die
erwarteten Zahlen stehen jeweils dabei – sie stammen aus der begleitenden
Python-Reproduktion (`analysis.py`) und sollten in Orange (bis auf
Zufallsschwankungen) gleich herauskommen.

> **Tipp zur Abgabe:** Speichere die fertige Orange-Datei als
> `diabetes.ows` und gib sie zusammen mit dem Word-Bericht ab. Mache von jedem
> wichtigen Widget einen Screenshot (Baum, Test & Score, Konfusionsmatrix) für
> den Bericht.

---

## 0. Vorbereitung

1. Orange installieren (orangedatamining.com) und öffnen → **New Workflow**.
2. Datei `data/diabetes_clean.csv` bereithalten (bereits bereinigt, s. u.).
   Wer die Reinigung in Orange selbst zeigen will, nimmt `data/diabetes_raw.csv`
   und führt Schritt 2 durch.

---

## 1. Daten laden  ·  Widget: **File**

- Widget **File** auf die Leinwand ziehen, `diabetes_raw.csv` laden.
- Doppelklick → Spalten prüfen. **Wichtige Rollen einstellen:**
  - `Outcome` → Rolle **target** (Zielvariable), Typ **categorical**.
  - Die acht Merkmale → Rolle **feature**, Typ **numeric**.
- Mit **Data Table** kontrollieren: 768 Zeilen, 9 Spalten.

## 2. Datenreinigung  ·  Widgets: **Impute** (und ggf. *Edit Domain*)

Die „unmöglichen Nullen" (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
sind in Wahrheit fehlende Werte.

- Variante A (empfohlen, einfach): Die mitgelieferte Datei
  `data/diabetes_clean.csv` verwenden – dort sind die Nullen bereits durch den
  **Median** ersetzt (Glucose→117, BloodPressure→72, SkinThickness→29,
  Insulin→125, BMI→32.3). Dann Schritt 2 überspringen.
- Variante B (Reinigung in Orange zeigen): Mit dem Widget **Impute** die fünf
  Spalten auf „Average/Most frequent" bzw. **Median** setzen. (Die Nullen vorher
  als fehlend markieren, z. B. über *Feature Constructor* / *Preprocess*.)

> Erwartung: keine Zeile wird gelöscht, weiterhin **768** Datensätze.

## 3. Explorative Datenanalyse  ·  Widgets: **Distributions**, **Box Plot**, **Scatter Plot**, **Correlations**

- **Distributions**: je Merkmal die Verteilung anzeigen, „Split by → Outcome".
  → Glucose trennt am deutlichsten.
- **Box Plot**: Variable Glucose bzw. BMI, „Subgroups → Outcome".
- **Correlations**: zeigt die Zusammenhänge. Erwartung: Glucose–Outcome ≈ **0.49**
  (stärkste), dann BMI ≈ 0.31, Age ≈ 0.24.
- Optional **Scatter Plot** (Glucose vs. BMI, Farbe = Outcome) für die
  Klassentrennung.

## 4. Train/Test-Aufteilung & Modelle  ·  Widgets: **Tree**, **kNN**, **Test & Score**

**Modelle definieren:**

- **Tree**-Widget mit diesen Parametern:
  - „Induce binary tree" = an
  - „Min. number of instances in leaves" = **5**
  - „Limit the maximal tree depth to" = **4**
  - (Kriterium Gini ist Standard)
- **kNN**-Widget: „Number of neighbors" = **5**, Metric = Euclidean.
  - Davor ein **Preprocess**-Widget mit **Normalize Features** (kNN ist
    distanzbasiert und braucht standardisierte Merkmale!).

**Auswerten mit Test & Score:**

- Widget **Test & Score** öffnen, Datenfluss: *File/Impute → Test & Score*,
  und *Tree → Test & Score*, *kNN → Test & Score*.
- Sampling-Methode: **Random sampling**, **70 % Training**, **stratified**.
- Angezeigte Metriken: AUC, CA (=Accuracy), F1, Precision, Recall.

> Erwartung (eine Beispiel-Aufteilung, Tree): CA ≈ 0.71, Precision ≈ 0.58,
> Recall ≈ 0.69, F1 ≈ 0.63, AUC ≈ 0.79.

**Konfusionsmatrix:** Widget **Confusion Matrix** an *Test & Score* hängen.
> Erwartung Tree (Beispiel): 109 / 41 / 25 / 56 (siehe Bericht, Abb. 6).

## 5. Baum visualisieren  ·  Widget: **Tree Viewer**

- *Tree → Tree Viewer*. **Wichtig:** Damit der gezeigte Baum exakt zur
  Gini-Handrechnung passt, das *Tree*-Widget **mit dem ganzen Datensatz** speisen
  (also *File/Impute → Tree → Tree Viewer*, ohne Data Sampler dazwischen).
- Erwartung: erste Entscheidung **Glucose ≤ 127.5**, Wurzel-Gini ≈ **0.454**,
  Wurzel-Werte **[500, 268]**. Screenshot für den Bericht machen.

## 6. Zehn zufällige Aufteilungen  ·  Widget: **Test & Score**

Pro Durchlauf in **Test & Score** unter „Random sampling" die Option
**„Repeat train/test"** auf **10** stellen – Orange mittelt dann automatisch über
10 zufällige 70/30-Aufteilungen. Die Einzelwerte je Aufteilung lassen sich auch
manuell sammeln (Sampler-Seed ändern). Werte in die Versuchstabelle übertragen
und **Mittelwert / Minimum / Maximum** berechnen (siehe `Versuchstabelle.csv`).

> Erwartung (Ø über 10, Tree): CA ≈ **0.731**, F1 ≈ **0.614**, AUC ≈ **0.791**.
> kNN: CA ≈ 0.728, F1 ≈ 0.599, AUC ≈ 0.779.

## 7. Gini von Hand verifizieren

Siehe Bericht, Abschnitt 2d / Schritt 3. Aus dem Tree Viewer die Knotenzahlen
ablesen und nachrechnen:

- Wurzel (768 = 500 + 268): Gini = **0.4544**
- `Glucose ≤ 127.5`: links 485 (391/94), rechts 283 (109/174) → gew. Gini **0.3719**
- Alternative `BMI ≤ 30`: links 292 (241/51), rechts 476 (259/217) → gew. Gini **0.4171**

→ Glucose hat das kleinere gewichtete Gini, deshalb wählt Orange es als erste
Entscheidung. Struktur bestätigt. ✓

---

## Übersicht des Widget-Graphen

```
                                 ┌─────────────┐
                           ┌────▶│ Tree Viewer │   (ganzer Datensatz: Baum + Gini)
                           │     └─────────────┘
 ┌──────┐   ┌─────────┐    │     ┌──────────────┐    ┌───────────────────┐
 │ File │──▶│ Impute  │────┼────▶│ Test & Score │───▶│ Confusion Matrix  │
 └──────┘   └─────────┘    │     └──────────────┘    └───────────────────┘
                           │        ▲       ▲
                           │   ┌────┘       └─────────┐
                           │ ┌──────┐            ┌──────────┐   ┌──────┐
                           └▶│ Tree │            │Preprocess│──▶│ kNN  │
                             └──────┘            │(Normalize)│   └──────┘
                                                 └──────────┘
                                                      ▲
                                                      │ (Daten)
```

(EDA-Widgets *Distributions / Box Plot / Correlations* hängen ebenfalls direkt
an *Impute*.)
