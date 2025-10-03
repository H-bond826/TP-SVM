# Projet d'Apprentissage Statistique - Machines à Vecteurs de Support (SVM)

---

### Réalisé par
*   **[Qingjian ZHU]**

*Projet pour le cours HAX907X - Année 2025-2026*

## 🎯 Objectif du Projet

Ce dépôt contient l'ensemble du travail réalisé pour le TP sur les Machines à Vecteurs de Support (SVM). Il inclut le code Python pour l'implémentation des modèles, l'analyse des données, ainsi que le rapport détaillé rédigé avec Quarto.

## 📂 Structure du Dépôt

Le projet est organisé selon la structure recommandée suivante pour une meilleure clarté :
```
├── report/                # Dossier contenant le rapport
│   ├── report.qmd         # Fichier source Quarto du rapport
│   └── report.pdf         # Rapport final généré au format PDF
│
├── src/                   # Dossier contenant les scripts Python
│   ├── svm_scripy.py            # Script principal d'exécution
│   └── svm_source.py      # Fonctions utilitaires (si applicable)
│
├── requirements.txt       # Liste des dépendances Python
├── .gitignore             # Fichiers ignorés par Git
└── README.md              # Ce fichier d'instructions
```

## 🚀 Guide de Démarrage et de Reproduction

Pour exécuter le code et générer le rapport final, veuillez suivre ces étapes.

### 1. Prérequis
> Assurez-vous que les logiciels suivants sont installés sur votre système :
> - **Python** (version 3.8 ou supérieure)
> - **Quarto CLI** ([Instructions d'installation](https://quarto.org/docs/get-started/))
> - Une distribution **LaTeX** (comme MiKTeX ou TeX Live) pour la compilation du PDF.

### 2. Installation de l'environnement

Clonez le dépôt et installez les dépendances Python dans un environnement virtuel.
```bash
# 1. Cloner le dépôt
git clone https://github.com/H-bond826/TP-SVM.git
cd TP-SVM

# 2. Créer et activer un environnement virtuel
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# 3. Installer les librairies requises
pip install -r requirements.txt
```

### 3. Génération du Rapport

Une fois l'environnement configuré, lancez la commande suivante depuis la racine du projet pour compiler le rapport :
```bash
quarto render report/report.qmd
```
