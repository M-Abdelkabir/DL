"""
app.py — Serveur Flask pour le chatbot RAG FSSM / UCA Marrakech
Lancement : python app.py  (ou  flask run)
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from models.rag_pipeline import get_rag_pipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_ENSURE_ASCII"] = False   # garder les accents en JSON

# ── Lazy-load du pipeline RAG (chargé une seule fois au premier appel) ────────
_rag = None

def get_pipeline():
    global _rag
    if _rag is None:
        log.info("Chargement du pipeline RAG…")
        _rag = get_rag_pipeline()
        log.info("Pipeline RAG prêt ✓")
    return _rag


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Page principale — interface chatbot."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body JSON  : { "query": "...", "top_k": 5 }   (top_k optionnel)
    Response   : { "answer": "...", "sources": [...] }
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Le champ 'query' est obligatoire."}), 400

    if len(query) > 1000:
        return jsonify({"error": "Question trop longue (max 1000 caractères)."}), 400

    try:
        top_k = int(data.get("top_k", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "Le champ 'top_k' doit être un entier."}), 400
    top_k = max(1, min(top_k, 10))   # borne entre 1 et 10

    log.info("Query : %s", query[:120])

    try:
        rag = get_pipeline()
        response = rag.answer(query, top_k=top_k)

        return jsonify({
            "answer":  response.answer,
            "sources": response.sources,
            "top_k":   response.top_k,
        })

    except Exception as exc:
        log.exception("Erreur pipeline RAG")
        return jsonify({"error": "Erreur interne du serveur. Veuillez réessayer."}), 500


@app.route("/api/health")
def health():
    """
    GET /api/health — endpoint de santé (utile pour monitoring / Docker).
    """
    return jsonify({
        "status":  "ok",
        "service": "FSSM Chatbot",
        "rag_loaded": _rag is not None,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    log.info("Démarrage FSSM Chatbot sur http://localhost:%d", port)

    # Pré-chargement du pipeline au démarrage (optionnel, évite la latence
    # au 1er appel utilisateur). Commenter si le modèle est lourd.
    try:
        get_pipeline()
    except Exception as e:
        log.warning("Pré-chargement pipeline échoué (%s) — chargé au 1er appel.", e)

    app.run(host="0.0.0.0", port=port, debug=debug)
