"""
RAWSM — Role-Aware Weighted Semantic Matching
Core scoring pipeline (framework-agnostic).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

BULK_PAGE_THRESHOLD = 5
BULK_WORD_THRESHOLD = 3000
MIN_CHAR_COUNT      = 100
MIN_WORD_COUNT      = 200
MIN_GROUP_SIZE      = 5
MAX_SECTION_WORDS   = 120
TOP_K               = 10

TECH_NORMALIZE: Dict[str, str] = {
    r"\bNLP\b"   : "natural language processing",
    r"\bML\b"    : "machine learning",
    r"\bDL\b"    : "deep learning",
    r"\bCV\b"    : "computer vision",
    r"\bLLM\b"   : "large language model",
    r"\bRAG\b"   : "retrieval augmented generation",
    r"\bXAI\b"   : "explainable artificial intelligence",
    r"\bAPI\b"   : "application programming interface",
    r"\bCI/CD\b" : "continuous integration continuous deployment",
    r"\bDBMS\b"  : "database management system",
    r"\bOOP\b"   : "object oriented programming",
    r"\bREST\b"  : "representational state transfer",
    r"\bMLOps\b" : "machine learning operations",
    r"\bFE\b"    : "frontend",
    r"\bBE\b"    : "backend",
}

SECTION_PATTERNS: Dict[str, str] = {
    "summary"       : r"(summary|objective|profile|about me|personal statement)",
    "experience"    : r"(experience|employment|work history|professional background|career)",
    "projects"      : r"(projects|portfolio|personal projects|key projects|academic projects)",
    "skills"        : r"(skills|technical skills|core competencies|technologies|tech stack|tools)",
    "education"     : r"(education|academic|qualifications|degrees|university|college)",
    "certifications": r"(certifications|certificates|courses|training|licenses|credentials)",
}

SECTION_ORDER = ["experience", "projects", "skills", "summary", "certifications", "education", "general"]

TIER_LABELS = {0: "Short (<200w)", 1: "Medium (200–500w)", 2: "Long (>500w)"}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    filename        : str
    name            : str
    word_count      : int
    n_pages         : int
    content_density : float
    ambiguous       : bool
    bulk_flag       : bool
    missing_flag    : bool
    length_tier     : int
    role_scores     : Dict[str, float]          = field(default_factory=dict)
    role_scores_norm: Dict[str, float]          = field(default_factory=dict)
    section_scores  : Dict[str, Dict[str, float]] = field(default_factory=dict)
    fl_adjusted     : Dict[str, bool]           = field(default_factory=dict)
    relevant        : Dict[str, bool]           = field(default_factory=dict)
    top_drivers     : Dict[str, List[str]]      = field(default_factory=dict)


@dataclass
class FairnessReport:
    role       : str
    dpd_before : Optional[float]
    dpd_after  : Optional[float]
    n_adjusted : int
    tier_rates : Dict[str, float] = field(default_factory=dict)


@dataclass
class RankingResult:
    role            : str
    ranked          : List[CandidateResult]
    precision_at_5  : float
    precision_at_10 : float
    fairness        : FairnessReport


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes, filename: str) -> Tuple[str, int]:
    """Extract plain text and page count from PDF bytes."""
    try:
        doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
        text   = " ".join(page.get_text() for page in doc)
        pages  = len(doc)
        doc.close()
        return text.strip(), pages
    except Exception:
        return "", 0


# ── Text preprocessing ────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    if not text or len(text.strip()) < 10:
        return ""

    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r"[â€™â€œâ€\x80-\x9f]", " ", text)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"(\+?\d[\d\s\-().]{7,}\d)", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"linkedin\.com/\S+", "", text)
    text = re.sub(r"github\.com/\S+", "", text)

    text = re.sub(r"[•●◦▪▸►✓✔–—]", " ", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*[|/\\=_*#~`]+\s*$", "", text, flags=re.MULTILINE)

    for pattern, replacement in TECH_NORMALIZE.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text.strip()


# ── Section detection ─────────────────────────────────────────────────────────

def detect_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {k: "" for k in SECTION_PATTERNS}
    sections["general"] = ""

    if not text:
        return sections

    lines, current, buffer = text.split("\n"), "general", []

    for line in lines:
        stripped, matched = line.strip(), False
        for section, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, stripped, re.IGNORECASE) and len(stripped) < 60:
                if buffer:
                    sections[current] += " ".join(buffer) + " "
                    buffer = []
                current, matched = section, True
                break
        if not matched and stripped:
            buffer.append(stripped)

    if buffer:
        sections[current] += " ".join(buffer)

    return {k: v.strip() for k, v in sections.items()}


# ── Scoring helpers ───────────────────────────────────────────────────────────

def content_density(sections: Dict[str, str], total_words: int) -> float:
    if total_words == 0:
        return 0.0
    named = sum(len(v.split()) for k, v in sections.items() if k in SECTION_PATTERNS and v)
    return min(named / total_words, 1.0)


def section_level_scores(
    sections: Dict[str, str],
    jd_emb,
    model: SentenceTransformer,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name, text in sections.items():
        if text and len(text.strip()) > 20:
            words   = text.split()
            trimmed = " ".join(words[:MAX_SECTION_WORDS])
            emb     = model.encode(trimmed, convert_to_tensor=True)
            scores[name] = float(util.cos_sim(emb, jd_emb))
        else:
            scores[name] = 0.0
    return scores


def weighted_score(sec_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total = total_w = 0.0
    for sec, w in weights.items():
        total   += sec_scores.get(sec, 0.0) * w
        total_w += w
    return total / total_w if total_w else 0.0


def length_tier(word_count: int) -> int:
    if word_count < 200:
        return 0
    elif word_count <= 500:
        return 1
    return 2


def top_section_drivers(sec_scores: Dict[str, float], n: int = 3) -> List[str]:
    return [s for s, v in sorted(sec_scores.items(), key=lambda x: x[1], reverse=True) if v > 0][:n]


def precision_at_k(candidates: List[CandidateResult], role: str, k: int) -> float:
    ranked = sorted(candidates, key=lambda c: c.role_scores_norm.get(role, 0), reverse=True)[:k]
    return sum(1 for c in ranked if c.relevant.get(role, False)) / k


# ── FairLearn calibration ─────────────────────────────────────────────────────

def run_fairlearn(
    candidates: List[CandidateResult],
    role: str,
) -> FairnessReport:
    col_norm  = f"{role}_norm"
    X         = np.array([[c.role_scores_norm[role]] for c in candidates])
    y         = (X.flatten() >= 0.5).astype(int)
    sensitive = np.array([c.length_tier for c in candidates])

    # Merge undersized tiers upward
    tier_counts = {t: int((sensitive == t).sum()) for t in np.unique(sensitive)}
    merged      = np.array([
        min(t + 1, 2) if tier_counts.get(t, 0) < MIN_GROUP_SIZE else t
        for t in sensitive
    ])

    n_groups  = len(np.unique(merged))
    n_classes = len(np.unique(y))

    if n_groups < 2 or n_classes < 2:
        for c in candidates:
            c.fl_adjusted[role] = False
        return FairnessReport(role=role, dpd_before=None, dpd_after=None, n_adjusted=0)

    # Validate each group has both label classes
    for grp in np.unique(merged):
        if len(np.unique(y[merged == grp])) < 2:
            for c in candidates:
                c.fl_adjusted[role] = False
            return FairnessReport(role=role, dpd_before=None, dpd_after=None, n_adjusted=0)

    base_clf    = LogisticRegression(max_iter=500)
    base_clf.fit(X, y)
    y_pred_base = base_clf.predict(X)
    dpd_before  = demographic_parity_difference(y, y_pred_base, sensitive_features=merged)

    try:
        optimizer = ThresholdOptimizer(
            estimator      = base_clf,
            constraints    = "demographic_parity",
            predict_method = "predict_proba",
            objective      = "balanced_accuracy_score",
        )
        optimizer.fit(X, y, sensitive_features=merged)
        adjusted = optimizer.predict(X, sensitive_features=merged)
    except Exception:
        adjusted = y_pred_base

    dpd_after  = demographic_parity_difference(y, adjusted, sensitive_features=merged)
    n_adjusted = int((adjusted != y_pred_base).sum())

    for i, c in enumerate(candidates):
        c.fl_adjusted[role] = bool(adjusted[i] != y_pred_base[i])

    mf = MetricFrame(
        metrics            = {"selection_rate": lambda yt, yp: yp.mean()},
        y_true             = y,
        y_pred             = adjusted,
        sensitive_features = merged,
    )
    tier_rates = {
        TIER_LABELS.get(int(t), str(t)): float(r)
        for t, r in mf.by_group["selection_rate"].items()
    }

    return FairnessReport(
        role       = role,
        dpd_before = dpd_before,
        dpd_after  = dpd_after,
        n_adjusted = n_adjusted,
        tier_rates = tier_rates,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

class RAWSMPipeline:
    def __init__(
        self,
        job_descriptions: Dict[str, str],
        role_weights: Dict[str, Dict[str, float]],
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.job_descriptions = job_descriptions
        self.role_weights     = role_weights
        self.model            = SentenceTransformer(model_name)
        self.jd_embeddings    = {
            role: self.model.encode(jd, convert_to_tensor=True)
            for role, jd in job_descriptions.items()
        }

    def process_cv(self, pdf_bytes: bytes, filename: str) -> Optional[CandidateResult]:
        """Extract, preprocess, and score a single CV. Returns None if bulk."""
        raw_text, n_pages = extract_pdf_text(pdf_bytes, filename)
        word_count        = len(raw_text.split())
        char_count        = len(raw_text)

        is_bulk    = n_pages > BULK_PAGE_THRESHOLD or word_count > BULK_WORD_THRESHOLD
        is_missing = char_count < MIN_CHAR_COUNT

        name       = filename.replace(".pdf", "").replace("_", " ").strip()
        clean_text = preprocess(raw_text)
        sections   = detect_sections(clean_text)
        density    = content_density(sections, word_count)
        tier       = length_tier(word_count)

        candidate = CandidateResult(
            filename        = filename,
            name            = name,
            word_count      = word_count,
            n_pages         = n_pages,
            content_density = density,
            ambiguous       = word_count < MIN_WORD_COUNT or density < 0.25,
            bulk_flag       = is_bulk,
            missing_flag    = is_missing,
            length_tier     = tier,
        )

        if is_bulk or is_missing:
            return candidate

        for role, jd_emb in self.jd_embeddings.items():
            sec_scores  = section_level_scores(sections, jd_emb, self.model)
            raw_score   = weighted_score(sec_scores, self.role_weights[role])
            penalised   = raw_score * (0.85 + 0.15 * density)

            candidate.section_scores[role] = sec_scores
            candidate.role_scores[role]    = penalised
            candidate.top_drivers[role]    = top_section_drivers(sec_scores)

        return candidate

    def rank(self, pdf_files: Dict[str, bytes]) -> Dict[str, RankingResult]:
        """
        Score and rank a batch of CVs against all roles.

        Args:
            pdf_files: {filename: pdf_bytes}

        Returns:
            {role: RankingResult}
        """
        # 1. Extract + score all CVs
        all_candidates = [
            self.process_cv(data, fname)
            for fname, data in pdf_files.items()
        ]
        all_candidates = [c for c in all_candidates if c is not None]

        # 2. Filter to scoreable candidates
        scoreable = [c for c in all_candidates if not c.bulk_flag and not c.missing_flag]

        if not scoreable:
            return {}

        # 3. Normalise scores per role
        scaler = MinMaxScaler()
        for role in self.job_descriptions:
            raw = np.array([[c.role_scores.get(role, 0.0)] for c in scoreable])
            norm = scaler.fit_transform(raw).flatten()
            for c, n in zip(scoreable, norm):
                c.role_scores_norm[role] = float(n)

        # 4. Relevance flags (60th percentile threshold)
        for role in self.job_descriptions:
            scores    = [c.role_scores_norm[role] for c in scoreable]
            threshold = float(np.percentile(scores, 60))
            for c in scoreable:
                c.relevant[role] = c.role_scores_norm[role] >= threshold

        # 5. FairLearn calibration per role
        fairness_reports: Dict[str, FairnessReport] = {}
        for role in self.job_descriptions:
            fairness_reports[role] = run_fairlearn(scoreable, role)

        # 6. Build ranked results per role
        results: Dict[str, RankingResult] = {}
        for role in self.job_descriptions:
            ranked = sorted(
                scoreable,
                key=lambda c: c.role_scores_norm.get(role, 0),
                reverse=True,
            )[:TOP_K]

            results[role] = RankingResult(
                role            = role,
                ranked          = ranked,
                precision_at_5  = precision_at_k(scoreable, role, 5),
                precision_at_10 = precision_at_k(scoreable, role, 10),
                fairness        = fairness_reports[role],
            )

        return results
