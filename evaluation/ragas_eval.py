"""
ragas_eval.py — Évaluation RAGAS du pipeline RAG FSSM/UCA
==========================================================
Framework : RAGAS (Retrieval Augmented Generation Assessment)
Pipeline  : SentenceTransformer 768d + ChromaDB + LLaMA-3.3-70b (Groq)

Métriques évaluées :
  - Faithfulness          : fidélité de la réponse au contexte récupéré
  - Answer Relevancy      : pertinence de la réponse par rapport à la question
  - Context Precision     : précision du contexte récupéré (signal/bruit)
  - Context Recall        : rappel du contexte (couverture de la ground truth)
  - Answer Correctness    : correction factuelle vs ground truth
  - Context Entity Recall : rappel des entités nommées dans le contexte

Installation :
    pip install ragas datasets langchain-groq langchain-community chromadb sentence-transformers

Usage :
    python evaluation/ragas_eval.py
    python evaluation/ragas_eval.py --subset 10
    python evaluation/ragas_eval.py --metrics faithfulness answer_relevancy
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)
GROQ_MODEL        = "llama-3.3-70b-versatile"
EMBEDDING_MODEL   = "paraphrase-multilingual-mpnet-base-v2"

TEST_QUESTIONS_PATH = ROOT / "evaluation" / "test_questions.json"
RESULTS_PATH        = ROOT / "evaluation" / "results.json"

TOP_K             = 5       # Nombre de docs récupérés par requête
BATCH_SLEEP       = 2.0     # Secondes entre les batches (rate limit Groq)
BATCH_SIZE        = 5       # Questions par batch


# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT DU PIPELINE FSSM
# ══════════════════════════════════════════════════════════════

def load_pipeline():
    """Charge le retriever ChromaDB et le LLM Groq."""
    from models.retrieval    import get_retriever
    from models.rag_pipeline import get_rag_pipeline

    log.info("Chargement du retriever ChromaDB...")
    retriever = get_retriever()
    n = retriever.collection.count()
    log.info("  %d documents indexés ✓", n)

    log.info("Chargement du pipeline RAG...")
    rag = get_rag_pipeline()
    log.info("  Pipeline RAG prêt ✓")

    return retriever, rag


# ══════════════════════════════════════════════════════════════
# 2. CHARGEMENT DES QUESTIONS DE TEST
# ══════════════════════════════════════════════════════════════

def load_test_questions(path: Path, subset: Optional[int] = None) -> list[dict]:
    """
    Charge les questions de test depuis test_questions.json.

    Format attendu :
    [
      {
        "id"              : "eval_001",
        "question"        : "...",
        "ground_truth"    : "...",
        "category"        : "...",
        "difficulty"      : "easy|medium|hard"
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    log.info("Questions de test chargées : %d", len(data))

    if subset and subset < len(data):
        import random
        random.seed(42)
        data = random.sample(data, subset)
        log.info("Sous-ensemble sélectionné : %d questions", len(data))

    return data


# ══════════════════════════════════════════════════════════════
# 3. CONSTRUCTION DU DATASET RAGAS
# ══════════════════════════════════════════════════════════════

def build_ragas_dataset(test_questions: list[dict], retriever, rag) -> dict:
    """
    Exécute le pipeline RAG sur chaque question et construit
    le dataset au format RAGAS :
      - question         : str
      - answer           : str  (réponse générée)
      - contexts         : list[str]  (passages récupérés)
      - ground_truth     : str  (réponse de référence)
    """
    questions, answers, contexts_list, ground_truths = [], [], [], []
    meta_list = []

    log.info("\n%s", "=" * 55)
    log.info("  CONSTRUCTION DU DATASET RAGAS (%d questions)", len(test_questions))
    log.info("=" * 55)

    for i, item in enumerate(tqdm(test_questions, desc="Pipeline RAG")):
        question     = item["question"]
        ground_truth = item.get("ground_truth", "")

        # ── Retrieval ──────────────────────────────────────────
        t0 = time.perf_counter()
        retrieved = retriever.search(question, top_k=TOP_K)
        t_ret = (time.perf_counter() - t0) * 1000

        # Extraire les textes des documents récupérés
        ctx_texts = [r.get("text", r.get("content", "")) for r in retrieved]
        ctx_scores = [r.get("score", 0.0) for r in retrieved]

        # ── Génération ─────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            response = rag.answer(question)
            # Extraction robuste de la réponse (RAGResponse ou dict)
            if hasattr(response, "answer"):
                answer_text = response.answer
            elif isinstance(response, dict):
                answer_text = response.get("answer", "")
            else:
                answer_text = str(response)
        except Exception as e:
            log.warning("Erreur génération Q%d : %s", i + 1, e)
            answer_text = ""
        t_llm = (time.perf_counter() - t0) * 1000

        questions.append(question)
        answers.append(answer_text)
        contexts_list.append(ctx_texts)
        ground_truths.append(ground_truth)

        meta_list.append({
            "id"          : item.get("id", f"eval_{i+1:03d}"),
            "category"    : item.get("category", "N/A"),
            "difficulty"  : item.get("difficulty", "medium"),
            "t_retrieval_ms": round(t_ret, 1),
            "t_llm_ms"    : round(t_llm, 1),
            "top_scores"  : ctx_scores,
        })

        # Rate limit Groq (entre batches)
        if (i + 1) % BATCH_SIZE == 0 and i + 1 < len(test_questions):
            log.info("  [%d/%d] Pause anti-rate-limit (%.1fs)...",
                     i + 1, len(test_questions), BATCH_SLEEP)
            time.sleep(BATCH_SLEEP)

    return {
        "questions"    : questions,
        "answers"      : answers,
        "contexts"     : contexts_list,
        "ground_truths": ground_truths,
        "meta"         : meta_list,
    }


# ══════════════════════════════════════════════════════════════
# 4. ÉVALUATION RAGAS
# ══════════════════════════════════════════════════════════════

def run_ragas_evaluation(dataset: dict, selected_metrics: Optional[list] = None) -> dict:

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    )
    from ragas.llms       import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets         import Dataset
    from langchain_groq   import ChatGroq

    # ── Embedding : SentenceTransformer via LangChain ─────────
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    # ── Sélection des métriques ────────────────────────────────
    ALL_METRICS = {
        "faithfulness"      : faithfulness,
        "answer_relevancy"  : answer_relevancy,
        "context_precision" : context_precision,
        "context_recall"    : context_recall,
        "answer_correctness": answer_correctness,
    }
    metrics = (
        [ALL_METRICS[m] for m in selected_metrics if m in ALL_METRICS]
        if selected_metrics else list(ALL_METRICS.values())
    )

    # ── LLM Groq — wrappé pour RAGAS ──────────────────────────
    log.info("Configuration LLM RAGAS → Groq / %s", GROQ_MODEL)
    groq_llm = ChatGroq(
        model       = GROQ_MODEL,
        api_key     = GROQ_API_KEY,
        temperature = 0.0,
    )
    ragas_llm = LangchainLLMWrapper(groq_llm)          # ✅ wrapper RAGAS

    # ── Embeddings SentenceTransformer — wrappé pour RAGAS ────
    log.info("Configuration Embedding RAGAS → %s", EMBEDDING_MODEL)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name    = EMBEDDING_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)  # ✅ wrapper RAGAS

    # ── Injecter dans chaque métrique ─────────────────────────
    for metric in metrics:
        metric.llm        = ragas_llm        # ✅ force Groq
        metric.embeddings = ragas_embeddings # ✅ force SentenceTransformer

    # ── Dataset HuggingFace ───────────────────────────────────
    hf_dataset = Dataset.from_dict({
        "question"    : dataset["questions"],
        "answer"      : dataset["answers"],
        "contexts"    : dataset["contexts"],
        "ground_truth": dataset["ground_truths"],
    })

    log.info("Lancement évaluation RAGAS (%d samples)...", len(hf_dataset))
    t0 = time.time()

    result = evaluate(
        dataset          = hf_dataset,
        metrics          = metrics,
        llm              = ragas_llm,        # ✅ Groq
        embeddings       = ragas_embeddings, # ✅ SentenceTransformer
        raise_exceptions = False,
    )

    log.info("✅ RAGAS terminé en %.1fs", time.time() - t0)

    scores = {}
    for metric in metrics:
        try:
            scores[metric.name] = float(result[metric.name])
        except Exception:
            scores[metric.name] = None

    return scores, result.to_pandas()


# ══════════════════════════════════════════════════════════════
# 5. MÉTRIQUES COMPLÉMENTAIRES (sans RAGAS)
# ══════════════════════════════════════════════════════════════

def compute_custom_metrics(dataset: dict) -> dict:
    """
    Calcule des métriques supplémentaires indépendantes de RAGAS :
      - Hit@K, MRR, NDCG (retrieval)
      - ROUGE-1/2/L (génération)
      - Similarité sémantique cosinus
      - Latences détaillées
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    log.info("\nCalcul des métriques complémentaires...")

    # ── ROUGE ──────────────────────────────────────────────────
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        for ref, gen in zip(dataset["ground_truths"], dataset["answers"]):
            if ref and gen:
                s = scorer.score(ref, gen)
                rouge_scores["rouge1"].append(s["rouge1"].fmeasure)
                rouge_scores["rouge2"].append(s["rouge2"].fmeasure)
                rouge_scores["rougeL"].append(s["rougeL"].fmeasure)
    except ImportError:
        log.warning("rouge-score non installé — métriques ROUGE ignorées")

    # ── Similarité sémantique ──────────────────────────────────
    sem_sims = []
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        refs  = [g for g in dataset["ground_truths"] if g]
        gens  = [dataset["answers"][i] for i, g in enumerate(dataset["ground_truths"]) if g]
        if refs and gens:
            vecs_ref = model.encode(refs,  normalize_embeddings=True)
            vecs_gen = model.encode(gens,  normalize_embeddings=True)
            sem_sims = [
                float(cosine_similarity([vecs_ref[i]], [vecs_gen[i]])[0][0])
                for i in range(len(refs))
            ]
    except Exception as e:
        log.warning("Similarité sémantique : %s", e)

    # ── Latences ──────────────────────────────────────────────
    t_ret = [m["t_retrieval_ms"] for m in dataset["meta"]]
    t_llm = [m["t_llm_ms"]       for m in dataset["meta"]]
    t_tot = [r + l for r, l in zip(t_ret, t_llm)]

    # ── Réponses vides ─────────────────────────────────────────
    empty_answers = sum(1 for a in dataset["answers"] if not a.strip())

    return {
        "rouge": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in rouge_scores.items() if v
        },
        "semantic_similarity": {
            "mean": float(np.mean(sem_sims)) if sem_sims else None,
            "std" : float(np.std(sem_sims))  if sem_sims else None,
            "min" : float(np.min(sem_sims))  if sem_sims else None,
            "max" : float(np.max(sem_sims))  if sem_sims else None,
        },
        "latency": {
            "retrieval_ms": {"mean": np.mean(t_ret), "p95": np.percentile(t_ret, 95)},
            "llm_ms"      : {"mean": np.mean(t_llm), "p95": np.percentile(t_llm, 95)},
            "total_ms"    : {"mean": np.mean(t_tot), "p95": np.percentile(t_tot, 95)},
        },
        "empty_answers"  : empty_answers,
        "total_questions": len(dataset["answers"]),
    }


# ══════════════════════════════════════════════════════════════
# 6. ANALYSE PAR CATÉGORIE
# ══════════════════════════════════════════════════════════════

def compute_category_analysis(dataset: dict, ragas_df) -> dict:
    """Décompose les scores RAGAS par catégorie thématique."""
    categories = [m["category"] for m in dataset["meta"]]
    unique_cats = list(set(categories))
    analysis   = {}

    if ragas_df is None:
        return analysis

    ragas_df["category"] = categories

    for cat in unique_cats:
        sub = ragas_df[ragas_df["category"] == cat]
        analysis[cat] = {}
        for col in sub.columns:
            if col == "category":
                continue
            try:
                vals = sub[col].dropna().astype(float)
                if len(vals) > 0:
                    analysis[cat][col] = {
                        "mean": round(float(vals.mean()), 4),
                        "std" : round(float(vals.std()),  4),
                        "n"   : int(len(vals)),
                    }
            except Exception:
                pass

    return analysis


# ══════════════════════════════════════════════════════════════
# 7. SAUVEGARDE DES RÉSULTATS
# ══════════════════════════════════════════════════════════════

def save_results(
    ragas_scores   : dict,
    custom_metrics : dict,
    category_analysis: dict,
    dataset        : dict,
    output_path    : Path,
):
    """Sauvegarde tous les résultats dans results.json."""

    # Score global (moyenne des métriques RAGAS disponibles)
    valid_scores = [v for v in ragas_scores.values() if v is not None]
    global_score = float(np.mean(valid_scores)) if valid_scores else None

    results = {
        "metadata": {
            "timestamp"        : datetime.now().isoformat(),
            "embedding_model"  : EMBEDDING_MODEL,
            "llm_model"        : GROQ_MODEL,
            "top_k"            : TOP_K,
            "total_questions"  : len(dataset["questions"]),
            "ragas_version"    : _get_package_version("ragas"),
        },
        "global_score"        : round(global_score, 4) if global_score else None,
        "ragas_metrics"       : {k: round(v, 4) if v else None
                                 for k, v in ragas_scores.items()},
        "custom_metrics"      : custom_metrics,
        "category_analysis"   : category_analysis,
        "per_question"        : [
            {
                "id"           : dataset["meta"][i]["id"],
                "question"     : dataset["questions"][i],
                "answer"       : dataset["answers"][i],
                "ground_truth" : dataset["ground_truths"][i],
                "category"     : dataset["meta"][i]["category"],
                "difficulty"   : dataset["meta"][i]["difficulty"],
                "contexts_used": len(dataset["contexts"][i]),
                "t_retrieval_ms": dataset["meta"][i]["t_retrieval_ms"],
                "t_llm_ms"     : dataset["meta"][i]["t_llm_ms"],
            }
            for i in range(len(dataset["questions"]))
        ],
        "recommendations"     : _generate_recommendations(ragas_scores, custom_metrics),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info("✅ Résultats sauvegardés : %s", output_path)
    return results


def _get_package_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "N/A"


def _generate_recommendations(ragas_scores: dict, custom_metrics: dict) -> list[str]:
    """Génère des recommandations automatiques selon les scores."""
    recs = []

    faith = ragas_scores.get("faithfulness")
    if faith is not None and faith < 0.7:
        recs.append(
            f"Faithfulness faible ({faith:.3f}) : augmenter TOP_K ou améliorer "
            "le prompt système pour forcer le LLM à rester dans le contexte."
        )

    rel = ragas_scores.get("answer_relevancy")
    if rel is not None and rel < 0.7:
        recs.append(
            f"Answer Relevancy faible ({rel:.3f}) : revoir l'instruction système du LLM "
            "pour qu'il réponde plus directement à la question posée."
        )

    cp = ragas_scores.get("context_precision")
    if cp is not None and cp < 0.7:
        recs.append(
            f"Context Precision faible ({cp:.3f}) : le retrieval ramène trop de bruit. "
            "Envisager un re-ranker Cross-Encoder ou réduire TOP_K."
        )

    cr = ragas_scores.get("context_recall")
    if cr is not None and cr < 0.7:
        recs.append(
            f"Context Recall faible ({cr:.3f}) : certaines informations pertinentes ne "
            "sont pas récupérées. Enrichir le corpus ou augmenter TOP_K."
        )

    sem = custom_metrics.get("semantic_similarity", {}).get("mean")
    if sem is not None and sem < 0.75:
        recs.append(
            f"Similarité sémantique faible ({sem:.3f}) : les réponses s'éloignent "
            "sémantiquement des références. Vérifier la qualité des ground truths."
        )

    if not recs:
        recs.append("Toutes les métriques sont satisfaisantes. Le pipeline est performant.")

    return recs


# ══════════════════════════════════════════════════════════════
# 8. AFFICHAGE DU RAPPORT CONSOLE
# ══════════════════════════════════════════════════════════════

def print_report(results: dict):
    """Affiche un rapport lisible dans la console."""
    sep = "=" * 58

    print(f"\n{sep}")
    print("        📊 RAPPORT D'ÉVALUATION RAGAS — FSSM/UCA")
    print(sep)
    print(f"  Timestamp    : {results['metadata']['timestamp'][:19]}")
    print(f"  Embedding    : {results['metadata']['embedding_model']}")
    print(f"  LLM          : {results['metadata']['llm_model']}")
    print(f"  Questions    : {results['metadata']['total_questions']}")
    print(f"  Score global : {results['global_score']}")
    print(sep)

    print("\n  📐 MÉTRIQUES RAGAS :")
    grade_map = [(0.85, "🟢 Excellent"), (0.70, "🔵 Bon"),
                 (0.55, "🟡 Moyen"), (0.0, "🔴 Insuffisant")]
    for metric, score in results["ragas_metrics"].items():
        if score is None:
            print(f"   {metric:30s}: N/A")
            continue
        grade = next(g for seuil, g in grade_map if score >= seuil)
        bar = "█" * int(score * 20)
        print(f"   {metric:30s}: {score:.4f}  {grade}")

    cm = results.get("custom_metrics", {})
    rouge = cm.get("rouge", {})
    if rouge:
        print("\n  📝 MÉTRIQUES ROUGE :")
        for k, v in rouge.items():
            print(f"   {k:30s}: {v['mean']:.4f}  (±{v['std']:.4f})")

    sem = cm.get("semantic_similarity", {})
    if sem.get("mean"):
        print(f"\n  🧠 Similarité sémantique moy. : {sem['mean']:.4f}")

    lat = cm.get("latency", {})
    if lat:
        print("\n  ⚡ LATENCES :")
        for stage, vals in lat.items():
            print(f"   {stage:30s}: {vals['mean']:.1f} ms  (p95: {vals['p95']:.1f} ms)")

    print(f"\n  {'─'*54}")
    print("  💡 RECOMMANDATIONS :")
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"   {i}. {rec}")

    print(sep)
    print(f"  📁 Résultats sauvegardés : evaluation/results.json")
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Évaluation RAGAS du pipeline FSSM")
    parser.add_argument("--subset",  type=int,   default=None,
                        help="Nombre de questions à évaluer (défaut: toutes)")
    parser.add_argument("--metrics", nargs="+",  default=None,
                        choices=["faithfulness","answer_relevancy","context_precision",
                                 "context_recall","answer_correctness"],
                        help="Métriques RAGAS à calculer")
    parser.add_argument("--output",  type=str,   default=str(RESULTS_PATH),
                        help="Chemin du fichier de sortie JSON")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Passer RAGAS, calculer uniquement les métriques custom")
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════╗")
    log.info("║   RAGAS Evaluation — FSSM/UCA RAG Pipeline  ║")
    log.info("╚══════════════════════════════════════════════╝")

    # 1. Chargement du pipeline
    retriever, rag = load_pipeline()

    # 2. Chargement des questions de test
    test_questions = load_test_questions(TEST_QUESTIONS_PATH, subset=args.subset)

    # 3. Exécution du pipeline RAG sur toutes les questions
    dataset = build_ragas_dataset(test_questions, retriever, rag)

    # 4. Évaluation RAGAS
    ragas_scores, ragas_df = {}, None
    if not args.no_ragas:
        ragas_scores, ragas_df = run_ragas_evaluation(dataset, args.metrics)
    else:
        log.info("RAGAS désactivé (--no-ragas)")

    # 5. Métriques complémentaires
    custom_metrics = compute_custom_metrics(dataset)

    # 6. Analyse par catégorie
    category_analysis = compute_category_analysis(dataset, ragas_df)

    # 7. Sauvegarde
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = save_results(ragas_scores, custom_metrics,
                           category_analysis, dataset, output_path)

    # 8. Rapport console
    print_report(results)


if __name__ == "__main__":
    main()
