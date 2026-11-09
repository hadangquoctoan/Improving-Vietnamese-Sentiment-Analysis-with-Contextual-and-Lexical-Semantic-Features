"""Model registry, preprocessing and per-word explanation for the demo API.

Preprocessing matches training (``scripts/prepare_vlsp2016.py``, ``scripts/predict_sentiment.py``):
NFC + lowercase + whitespace, VnCoreNLP word segmentation, then the same normalization again.
The teencode variant map and elongation collapse are NOT applied, because training did not use them.

The forward pass is re-run step by step (PhoBERT -> lexicon branch -> gate -> residual -> pooling ->
classifier) so the response can expose per-word gate values. ``tests/test_service.py`` checks that the
resulting logits equal ``SentimentModel.forward`` exactly.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from vnsentiment.models.fusion.pooling import pool_masked_mean
from vnsentiment.models.fusion.residual import fuse_residual
from vnsentiment.preprocessing.collapse import normalize_basic
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter
from vnsentiment.training.factory import build_sentiment_model

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"
LABELS_3 = (("neg", "Tiêu cực"), ("neu", "Trung tính"), ("pos", "Tích cực"))
LABELS_2 = (("neg", "Tiêu cực"), ("pos", "Tích cực"))  # AIVIVN: negative=0, positive=1
MAX_CHARS = 1000
MAX_LOADED = 3  # each model holds ~520 MB of VRAM; least recently used is unloaded beyond this

# Sentences run through the real UIT checkpoint on 2026-11-09 (see demo/README.md).
SAMPLES = (
    "Thầy cô hướng dẫn rất tận tâm",
    "Sản phẩm không tệ lắm",
    "Căng tin đông quá trời, chờ mãi không tới lượt",
    "Môn học này bình thường, không có gì đặc biệt",
    "Wifi trong trường quá yếu, không học online được",
    "Giảng viên dạy rất nhiệt tình nhưng phòng học hơi nóng",
)


_SEGMENTER: VnCoreNLPSegmenter | None = None


def _shared_segmenter(max_heap_size: str) -> VnCoreNLPSegmenter:
    """One VnCoreNLP per process: pyjnius can start the JVM only once, so a second instance would fail."""
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = VnCoreNLPSegmenter(max_heap_size=max_heap_size)
    return _SEGMENTER


class InputError(ValueError):
    """The request text or options cannot be processed; message is shown to the user."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    checkpoint: str
    dataset: str
    lexicon_version: str
    labels: tuple = LABELS_3
    recommended: bool = True  # shown in the demo picker; others are kept for comparison


# Provenance per plans-and-report session logs; all five verified on the demo machine (SHA/dev metrics).
SPECS = {
    "uit": ModelSpec(
        "uit", "UIT-VSFC", "Phản hồi sinh viên, 3 nhãn",
        "best_fused.pt", "UIT-VSFC", "v0.2.0",
    ),
    "vlsp": ModelSpec(
        "vlsp", "VLSP2016", "3 nhãn, huấn luyện lại từ đầu trên VLSP2016",
        "vlsp2016_scratch_best.pt", "VLSP2016", "v0.2.0",
    ),
    "aivivn": ModelSpec(
        "aivivn", "AIVIVN 2019", "2 nhãn: tiêu cực / tích cực",
        "aivivn2019_best.pt", "AIVIVN 2019", "v0.2.0", labels=LABELS_2,
    ),
    "uit-alt": ModelSpec(
        "uit-alt", "UIT-VSFC (bản 2)", "Bản train khác của nhóm; dev cao hơn nhưng chưa chạy test UIT",
        "best.pt", "UIT-VSFC", "v0.2.0", recommended=False,
    ),
    # Same recipe and seed as vlsp2016_scratch_best.pt (both start from pretrained PhoBERT, not from best.pt);
    # the scratch run reproduced it, so predictions match to 6 decimals although the bytes differ.
    "vlsp-run1": ModelSpec(
        "vlsp-run1", "VLSP2016 (lần chạy 1)", "Lần train đầu trên VLSP2016; bản huấn luyện lại cho kết quả trùng",
        "vlsp2016_best.pt", "VLSP2016", "v0.2.0", recommended=False,
    ),
}


class DemoService:
    def __init__(
        self,
        device: str | None = None,
        preload: tuple[str, ...] = ("uit",),
        max_heap_size: str = "-Xmx512m",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models: OrderedDict[str, torch.nn.Module] = OrderedDict()  # LRU order
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._segmenter = _shared_segmenter(max_heap_size)
        for model_id in preload:
            self._load(model_id)
        self._segmenter.segment("khởi động")  # start the JVM now, not on the first request
        if preload:
            self.predict(SAMPLES[0], preload[0])  # warm CUDA kernels

    # ---- registry -------------------------------------------------------
    def models(self) -> list[dict]:
        return [
            {
                "id": spec.id,
                "name": spec.name,
                "description": spec.description,
                "labels": [key for key, _ in spec.labels],
                "recommended": spec.recommended,
                "available": (CHECKPOINT_DIR / spec.checkpoint).is_file(),
                "loaded": spec.id in self._models,
            }
            for spec in SPECS.values()
        ]

    def loaded(self) -> list[str]:
        return list(self._models)

    def _load(self, model_id: str) -> torch.nn.Module:
        spec = SPECS[model_id]
        path = CHECKPOINT_DIR / spec.checkpoint
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu checkpoint {spec.checkpoint} trong {CHECKPOINT_DIR}")
        while len(self._models) >= MAX_LOADED:
            evicted, _ = self._models.popitem(last=False)
            self._meta.pop(evicted, None)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        payload = torch.load(path, map_location=self.device, weights_only=True)
        model = build_sentiment_model(self.device, num_labels=len(spec.labels))
        model.load_state_dict(payload["model_state"])
        model.eval()
        self._models[model_id] = model
        self._meta[model_id] = {
            "epoch": payload.get("epoch"),
            "dev_metrics": payload.get("dev_metrics", {}),
        }
        return model

    # ---- preprocessing --------------------------------------------------
    def prepare(self, text: object) -> str:
        if not isinstance(text, str):
            raise InputError("Câu nhập phải là chuỗi ký tự.")
        raw = text.strip()
        if not raw:
            raise InputError("Nhập một câu tiếng Việt để phân tích.")
        if len(raw) > MAX_CHARS:
            raise InputError(f"Câu dài {len(raw)} ký tự, vượt giới hạn {MAX_CHARS} ký tự.")
        normalized = normalize_basic(raw)
        segmented = normalize_basic(self._segmenter.segment(normalized))
        if not segmented:
            raise InputError("Câu không còn nội dung sau khi chuẩn hóa.")
        return segmented

    # ---- inference ------------------------------------------------------
    @staticmethod
    def forward(model, segmented: str, lexicon: bool):
        """Step-by-step forward that mirrors SentimentModel.forward and keeps intermediates."""
        batch, hidden = model.phobert.encode([segmented])
        if lexicon:
            lexical = model.lexicon([segmented], batch)
            gates = model.gate(hidden, lexical)
            pooled = pool_masked_mean(fuse_residual(hidden, gates, lexical), batch.content_mask)
        else:
            lexical = gates = None
            pooled = pool_masked_mean(hidden, batch.content_mask)
        return model.classifier(pooled), batch, hidden, lexical, gates

    def predict(self, text: object, model_id: str = "uit", lexicon: bool = True) -> dict:
        if model_id not in SPECS:
            raise InputError(f"Không có mô hình '{model_id}'. Chọn một trong: {', '.join(SPECS)}.")
        started = time.perf_counter()
        spec = SPECS[model_id]
        with self._lock:  # VnCoreNLP (JVM) and the GPU model are used by one request at a time
            model = self._models.get(model_id) or self._load(model_id)
            self._models.move_to_end(model_id)  # most recently used
            meta = self._meta[model_id]  # read under the lock: another request may evict it later
            segmented = self.prepare(text)
            try:
                with torch.no_grad():
                    logits, batch, hidden, lexical, gates = self.forward(model, segmented, lexicon)
            except ValueError as error:  # e.g. a literal <s> token typed by the user
                raise InputError(f"Không xử lý được câu này: {error}") from error
            probs = logits.softmax(dim=-1)[0].float().cpu().tolist()
            tokens, truncated = self._explain(model, segmented, batch, hidden, lexical, gates)

        label_id = max(range(len(probs)), key=probs.__getitem__)
        labels = spec.labels
        return {
            "label": labels[label_id][1],
            "label_key": labels[label_id][0],
            "label_id": label_id,
            "probs": {key: round(p, 6) for (key, _), p in zip(labels, probs)},
            "text": text.strip(),
            "segmented": segmented,
            "truncated": truncated,
            "lexicon": lexicon,
            "tokens": tokens,
            "coverage": {
                "words": len(tokens),
                "in_lexicon": sum(1 for token in tokens if token["in_lexicon"]),
            },
            "model": {
                "id": spec.id,
                "name": spec.name,
                "checkpoint": spec.checkpoint,
                "dataset": spec.dataset,
                "lexicon_version": spec.lexicon_version,
                "epoch": meta["epoch"],
                "dev_macro_f1": meta["dev_metrics"].get("macro_f1"),
            },
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }

    @staticmethod
    def _explain(model, segmented, batch, hidden, lexical, gates) -> tuple[list[dict], bool]:
        words = segmented.split()
        positions: dict[int, list[int]] = {}
        for position, word_index in enumerate(batch.word_ids[0].tolist()):
            if word_index >= 0:
                positions.setdefault(word_index, []).append(position)

        store = model.lexicon.store
        tokens = []
        for index, word in enumerate(words):
            subwords = positions.get(index, [])
            hit = store.lookup(word)
            in_lexicon = hit is not None and any(hit[1])
            dims = None
            if in_lexicon:
                values, mask = hit
                dims = {
                    # polarity is stored rescaled to [0, 1]; report it on its original [-1, 1] scale
                    name: None if not present else round(2 * value - 1 if k == 0 else value, 4)
                    for k, (name, value, present) in enumerate(zip(store.features, values, mask))
                }
            gate = ratio = None
            if subwords and gates is not None and in_lexicon:
                g = gates[0, subwords]
                gate = round(float(g.mean()), 4)
                injected = (g * lexical[0, subwords]).norm(dim=-1)
                context = hidden[0, subwords].norm(dim=-1).clamp_min(1e-6)
                ratio = round(float((injected / context).mean()), 4)
            tokens.append({
                "word": word,
                "in_lexicon": in_lexicon,
                "gate": gate,
                "lexicon_ratio": ratio,
                "dims": dims,
                "subwords": len(subwords),
            })
        truncated = any(not positions.get(index) for index in range(len(words)))
        return tokens, truncated
