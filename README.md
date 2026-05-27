# FSSM Assistant RAG

Assistant web Flask pour répondre aux questions sur la FSSM/UCA avec une pipeline RAG.

## Installation légère

Pour tester le serveur Flask, l'endpoint de santé et les tests unitaires:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m unittest discover -s tests -v
```

## Installation complète RAG

Pour utiliser la recherche sémantique et le modèle d'embeddings:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Sous Linux, `torch` peut télécharger de gros paquets CUDA. Pour une installation CPU plus légère, installez d'abord PyTorch CPU depuis les instructions officielles de PyTorch, puis installez le reste des dépendances.

## Docker CPU

Copiez l'exemple d'environnement puis ajoutez votre clé Groq:

```bash
cp .env.example .env
```

Construction et lancement:

```bash
docker compose up --build
```

L'application sera disponible sur:

```text
http://localhost:5000
```

L'image Docker installe PyTorch avec les roues CPU uniquement, afin d'éviter les très gros paquets CUDA/NVIDIA.

## Configuration

La clé Groq doit être définie dans l'environnement:

```bash
export GROQ_API_KEY="votre-cle"
```

Lancement:

```bash
.venv/bin/python app.py
```
