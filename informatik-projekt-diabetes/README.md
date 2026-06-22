# Informatik-Projekt: Diabetes-Früherkennung mit einem Entscheidungsbaum

Datenanalyse-Projekt für die **Informatik-Leistungsbeurteilung 2 (FS 2026)**.
Umgesetzt mit **Orange Data Mining**; alle Zahlen sind mit Python reproduzierbar.

## Fragestellung
Lässt sich allein aus einfachen Gesundheitskennzahlen (Blutzucker, BMI, Alter …)
mit einem Entscheidungsbaum vorhersagen, ob eine Person Diabetes hat – und taugt
ein solches Modell als günstiger Screening-Vorfilter?

## Datensatz
**Pima Indians Diabetes Database** (Kaggle / UCI). 768 Personen, 8 Merkmale,
binäre Zielvariable `Outcome` (0 = kein Diabetes, 1 = Diabetes).

## Aufbau des Ordners
```
informatik-projekt-diabetes/
├── README.md                         ← diese Datei
├── analysis.py                       ← reproduziert alle Zahlen & Abbildungen
├── data/
│   ├── diabetes_raw.csv              ← Rohdaten (768 × 9)
│   ├── diabetes_clean.csv            ← bereinigt (Nullen→Median) – für Orange
│   ├── eda_describe.csv              ← Kennzahlen je Merkmal
│   ├── zehn_splits.csv               ← Metriken der 10 Aufteilungen
│   └── results.json                  ← alle Ergebnisse maschinenlesbar
├── figures/                          ← 8 Abbildungen (PNG) für den Bericht
├── orange/
│   ├── Orange-Workflow-Anleitung.md  ← Schritt-für-Schritt in Orange
│   └── Versuchstabelle.csv           ← 10 Splits + Mittelwert/Min/Max
└── report/
    └── Bericht.md                    ← vollständiger Berichtsentwurf (DE)
```

## Wichtigste Ergebnisse (Ø über 10 Train/Test-Aufteilungen)
| Modell | Accuracy | F1 | AUC |
|---|---|---|---|
| Decision Tree (Tiefe 4) | **0.731** | **0.614** | **0.791** |
| kNN (k = 5, normiert) | 0.728 | 0.599 | 0.779 |

- Wichtigstes Merkmal: **Glucose** (erste Baum-Entscheidung: `Glucose ≤ 127.5`).
- Schwachstelle: niedriger **Recall** → das Modell übersieht zu viele echte
  Erkrankte (kritischste Fehlerart bei einem Screening).
- Gini-Handrechnung bestätigt die Baumstruktur: gew. Gini `Glucose ≤ 127.5`
  = **0.3719** < `BMI ≤ 30` = **0.4171** < Wurzel **0.4544**.

## Reproduzieren
```bash
pip install scikit-learn pandas matplotlib
python analysis.py        # vom Repo-Wurzelverzeichnis aus ausführen
```

## Hinweis zu generativen Hilfsmitteln
Bei der Erstellung wurde ein KI-Sprachmodell als Hilfsmittel verwendet
(Datensatzrecherche, Aufbau der Analyse, Strukturierung). Der Einsatz ist gemäss
Aufgabenstellung **vorgängig mit der Lehrperson abzusprechen** und im Bericht
offenzulegen (siehe `report/Bericht.md`, Abschnitt „Quellen & Hilfsmittel").
Inhaltliche Verantwortung liegt bei der/dem Studierenden.
