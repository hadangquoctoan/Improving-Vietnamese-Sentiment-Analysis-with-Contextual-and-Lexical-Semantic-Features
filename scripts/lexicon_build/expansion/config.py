"""Static configuration for the offline lexicon expansion build."""

from pathlib import Path

FEATURES: tuple[str, ...] = (
    "polarity",
    "valence",
    "arousal",
    "dominance",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "trust",
    "anticipation",
)

PARENT_VERSION = "0.2.0"
VERSION = "0.3.0"

ROOT = Path(__file__).resolve().parents[3]
BASE_PATH = (
    ROOT / "data" / "lexicons" / "composite_12d" / f"v{PARENT_VERSION}" / "composite_lexicon.json"
)
BASE_CONFLICTS_PATH = (
    ROOT / "data" / "lexicons" / "composite_12d" / f"v{PARENT_VERSION}" / "conflicts_vswn.tsv"
)
WORDNET_ROOT = ROOT / "data" / "lexicons" / "vietnamese-wordnet" / "extracted"
FASTTEXT_PATH = ROOT / "data" / "lexicons" / "fasttext" / "cc.vi.300.vec"
TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
DEV_PATH = ROOT / "data" / "processed" / "dev.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "lexicons" / "composite_12d" / f"v{VERSION}"
DEFAULT_CACHE_PATH = ROOT / "artifacts" / "cache" / "lexicon_expansion" / "wordnet_terms.json"

ALLOWED_FILES: tuple[str, ...] = ("adj", "verb.emotion", "noun.feeling", "noun.motive")
COVERAGE_FILES: tuple[str, ...] = ("adj", "adv", "verb.emotion")

SEED_MIN_ABS = 0.5

HARD_STOPWORDS: tuple[str, ...] = (
    "rất", "quá", "hơi", "lắm", "khá", "cũng", "vẫn", "đã", "đang", "sẽ",
    "vừa", "mới", "từng", "chưa", "không", "chẳng", "chả", "có", "là", "và",
    "với", "của", "cho", "để", "làm", "thì", "mà", "nên", "nhưng", "hoặc",
    "nếu", "vì", "do", "bởi", "tại", "từ", "đến", "ra", "vào", "lên",
    "xuống", "đi", "lại", "nữa", "được", "bị", "phải", "cần", "muốn",
    "một", "ít", "trước", "sau", "cuối", "đầu", "khác", "sát", "qua",
    "cùng", "nguyên", "tiền", "chi_tiết",
)

PRONOUN_STOPWORDS: tuple[str, ...] = (
    "em", "anh", "chị", "ông", "bà", "cô", "chú", "bác", "bạn", "tôi",
    "ta", "mình", "nó", "họ", "chúng", "ai", "gì", "nào", "sao", "thế",
    "vậy", "đây", "đó", "kia", "ấy", "ạ", "nhé", "nha", "chứ", "nhỉ",
)

VAD_NEUTRAL = 0.5
VAD_MARGIN = 0.1

FASTTEXT_DEFAULTS: dict[str, float | int] = {
    "top_k": 10,
    "min_cosine": 0.55,
    "min_sign_agreement": 0.8,
    "min_neighbors": 2,
    "min_mean_sim": 0.65,
    "shrink": 0.5,
    "min_abs_score": 0.15,
}

SOURCE_PRIORITY: dict[str, int] = {"wordnet_syn": 0, "fasttext_nn": 1}

REASONS: tuple[str, ...] = (
    "sign_conflict",
    "multi_group_disagreement",
    "audited_conflict",
    "audited_override",
    "stoplist",
    "low_similarity",
    "low_sign_agreement",
    "low_neighbors",
    "low_mean_sim",
    "low_score",
    "no_vector",
    "vad_conflict",
    "already_active",
    "source_precedence",
    "not_in_group",
)
