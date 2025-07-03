# MNIST Backend - API de Serving

Ce projet contient une API FastAPI conçue pour servir un modèle de reconnaissance de chiffres manuscrits (MNIST).

## 🎯 Rôle dans l'architecture MLOps

Ce service constitue la brique de **Serving/Inference** de notre architecture. Il **NE GÈRE PAS** l'entraînement du modèle. Sa responsabilité unique est de :

1. **Charger** un artefact de modèle pré-entraîné (`convnet.pt`)
2. **Exposer** une API REST pour la prédiction en temps réel
3. **Préprocesser** les images uploadées
4. **Retourner** les prédictions avec confiance

> ⚠️ **Important** : L'entraînement du modèle est géré par le projet `kedro` avec des pipelines MLOps structurés.

## 🏗️ Architecture

```
📦 mnist-backend/           # 🎯 SERVING UNIQUEMENT
├── src/api/               # API FastAPI
├── src/models/            # Définitions de modèles (pas d'entraînement)
├── models/                # Artefacts de modèles pré-entraînés
└── tests/                 # Tests de l'API

📦 kedro/                  # 🏋️ ENTRAÎNEMENT UNIQUEMENT  
├── pipelines/             # Pipelines Kedro (data + training)
├── conf/                  # Configuration MLOps
└── data/                  # Données et artifacts MLflow
```

## 🔄 Workflow de Production

1. **Entraînement** : `kedro run` (dans le projet kedro)
2. **Artefacts** : Modèle sauvegardé dans `kedro/data/`
3. **Copie** : Modèle copié vers `mnist-backend/models/`
4. **Serving** : API chargée et prête pour les prédictions

## 🚀 Démarrage rapide

### Prérequis
Assurez-vous qu'un modèle entraîné existe dans `models/convnet.pt`. 
Si ce n'est pas le cas, entraînez d'abord avec Kedro :

```bash
# Dans le projet kedro/
kedro run
```

### Lancement de l'API

```bash
# Installation des dépendances
pip install -r requirements.txt

# Lancement de l'API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Endpoints

- `POST /api/v1/predict` - Classification d'image (upload fichier)
- `GET /health` - Statut de l'API  
- `GET /api/info` - Informations sur l'API et le modèle
- `GET /docs` - Documentation Swagger interactive
- `GET /redoc` - Documentation ReDoc

## 🐳 Docker

```bash
# Build
docker build -t mnist-backend .

# Run
docker run -p 8000:8000 mnist-backend

# Avec volume pour modèles
docker run -p 8000:8000 -v $(pwd)/models:/app/models mnist-backend
```

## 📁 Structure du Projet

```
src/
├── api/
│   └── main.py          # 🚀 API FastAPI
└── models/
    ├── __init__.py
    └── convnet.py       # 🧠 Définition du modèle ConvNet
models/
└── convnet.pt          # 💾 Modèle pré-entraîné (depuis Kedro)
tests/
├── unit/                # 🧪 Tests unitaires
├── integration/         # 🔗 Tests d'intégration  
└── model_validation/    # ✅ Validation de modèle
```

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/unit/

# Tests d'intégration 
pytest tests/integration/

# Validation de modèle
pytest tests/model_validation/

# Tous les tests
pytest
```

## 🔧 CI/CD Pipeline

Le workflow GitHub Actions automatise :

1. **Tests** : Qualité de code (Black, Pylint, Flake8)
2. **Build** : Image Docker avec modèle
3. **Push** : Publication sur GitHub Container Registry
4. **Deploy** : Prêt pour déploiement via `mnist-deployment`

## 🤝 Intégration avec Kedro

Pour utiliser un nouveau modèle entraîné :

```bash
# 1. Entraîner avec Kedro
cd ../kedro && kedro run

# 2. Copier le modèle
cp kedro/data/06_models/convnet.pt mnist-backend/models/

# 3. Redémarrer l'API
# L'API chargera automatiquement le nouveau modèle
``` 