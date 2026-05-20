import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data"

# CHUNK_PROFILE branches the chunk-dependent artifact dirs (corpus, index,
# runs, plots) so multiple chunkings can coexist. Default empty preserves the
# legacy paths. DATASET_DIR stays unsuffixed because gold cites, questions,
# and llm_validated_quotes are chunking-independent.
CHUNK_PROFILE = os.environ.get("CHUNK_PROFILE", "").strip()
_PROFILE_SUFFIX = f"_{CHUNK_PROFILE}" if CHUNK_PROFILE else ""

DATASET_DIR = DATA_ROOT / "dataset"
CORPUS_DIR = DATA_ROOT / f"corpus{_PROFILE_SUFFIX}"
INDEX_DIR = DATA_ROOT / f"index{_PROFILE_SUFFIX}"
RUNS_DIR = DATA_ROOT / f"runs{_PROFILE_SUFFIX}"
VALIDATION_DIR = DATA_ROOT / "validation"

HF_DATASET_NAME = "reglab/legal_rag_hallucinations"

EMBEDDING_MODEL = "joe32140/ModernBERT-base-msmarco"
EMBEDDING_DIM = 768
EMBEDDING_MAX_SEQ_LEN = 8192

CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "100"))

BM25_TOPK = 50
DENSE_TOPK = 50
RRF_K = 60
FINAL_TOPK = 5

GENERATOR_MODEL_ID = "eu.anthropic.claude-sonnet-4-6" #"eu.anthropic.claude-haiku-4-5-20251001-v1:0"
JUDGE_MODEL_ID = "eu.anthropic.claude-opus-4-6-v1"
BEDROCK_REGION = "eu-west-3"

COURTLISTENER_API_BASE = "https://www.courtlistener.com/api/rest/v4"
COURTLISTENER_BULK_BASE = "https://www.courtlistener.com/api/bulk-data"
FEDERAL_COURTS = ["scotus"] + [f"ca{n}" for n in range(1, 12)] + ["cadc", "cafc"]
DEFAULT_DISTRACTOR_COUNT = 5_000

PLOTS_DIR = DATA_ROOT / f"plots{_PROFILE_SUFFIX}"

FEATURE_PREDICTOR_MODEL = (
    Path(__file__).parent.parent
    / "ottp_topology_analysis" / "outputs" / "model_v4_summary" / "model_v4_summary.pt"
)

DEFAULT_SEED = 42

for d in [DATA_ROOT, DATASET_DIR, CORPUS_DIR, INDEX_DIR, RUNS_DIR, VALIDATION_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
