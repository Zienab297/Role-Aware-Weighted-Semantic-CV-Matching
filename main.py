"""
RAWSM — FastAPI application.

Endpoints
─────────
POST /rank          Upload multiple CV PDFs, get ranked results per role.
GET  /health        Liveness check.
GET  /roles         List configured roles and their section weights.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import JOB_DESCRIPTIONS, ROLE_WEIGHTS
from app.pipeline import (
    CandidateResult,
    FairnessReport,
    RAWSMPipeline,
    RankingResult,
    TIER_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ── App lifespan — load model once at startup ─────────────────────────────────

pipeline: Optional[RAWSMPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    log.info("Loading RAWSM pipeline and embedding model...")
    pipeline = RAWSMPipeline(
        job_descriptions = JOB_DESCRIPTIONS,
        role_weights     = ROLE_WEIGHTS,
    )
    log.info("Pipeline ready.")
    yield
    pipeline = None

app = FastAPI(
    title       = "RAWSM — Role-Aware Weighted Semantic Matching",
    description = "Fairness-aware CV screening API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Response schemas ──────────────────────────────────────────────────────────

class SectionScores(BaseModel):
    experience     : float = 0.0
    projects       : float = 0.0
    skills         : float = 0.0
    summary        : float = 0.0
    certifications : float = 0.0
    education      : float = 0.0
    general        : float = 0.0


class CandidateResponse(BaseModel):
    rank            : int
    name            : str
    filename        : str
    score           : float
    content_density : float
    word_count      : int
    section_scores  : Dict[str, float]
    top_drivers     : List[str]
    relevant        : bool
    ambiguous       : bool
    fl_adjusted     : bool
    flags           : List[str]


class FairnessResponse(BaseModel):
    dpd_before  : Optional[float]
    dpd_after   : Optional[float]
    n_adjusted  : int
    tier_rates  : Dict[str, float]
    note        : str


class RoleResult(BaseModel):
    role            : str
    precision_at_5  : float
    precision_at_10 : float
    fairness        : FairnessResponse
    candidates      : List[CandidateResponse]


class RankResponse(BaseModel):
    total_uploaded : int
    total_scored   : int
    bulk_excluded  : int
    missing_excluded: int
    results        : Dict[str, RoleResult]


class RoleConfig(BaseModel):
    name    : str
    weights : Dict[str, float]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_flags(c: CandidateResult, role: str) -> List[str]:
    flags = []
    if c.ambiguous:
        flags.append(f"Short/low-density CV ({c.word_count} words, density={c.content_density:.2f})")
    if c.fl_adjusted.get(role, False):
        flags.append("FairLearn adjusted")
    return flags


def build_fairness_response(f: FairnessReport) -> FairnessResponse:
    if f.dpd_before is None:
        note = "Skipped — insufficient group or class diversity in this batch."
    elif f.dpd_after <= f.dpd_before:
        reduction = (f.dpd_before - f.dpd_after) / f.dpd_before * 100 if f.dpd_before else 0
        note = f"DPD reduced by {reduction:.1f}% after adjustment."
    else:
        note = (
            "DPD increased slightly — base classifier was near-zero bias; "
            "adjustment introduced a small opposite disparity (expected trade-off)."
        )
    return FairnessResponse(
        dpd_before = round(f.dpd_before, 4) if f.dpd_before is not None else None,
        dpd_after  = round(f.dpd_after,  4) if f.dpd_after  is not None else None,
        n_adjusted = f.n_adjusted,
        tier_rates = {k: round(v, 3) for k, v in f.tier_rates.items()},
        note       = note,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/roles", response_model=List[RoleConfig])
def list_roles():
    return [RoleConfig(name=role, weights=weights) for role, weights in ROLE_WEIGHTS.items()]


@app.post("/rank", response_model=RankResponse)
async def rank_cvs(files: List[UploadFile] = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Read all uploaded PDFs
    pdf_files: Dict[str, bytes] = {}
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF.")
        pdf_files[f.filename] = await f.read()

    log.info(f"Received {len(pdf_files)} CVs for ranking.")

    # Run pipeline
    ranking: Dict[str, RankingResult] = pipeline.rank(pdf_files)

    if not ranking:
        raise HTTPException(status_code=422, detail="No scoreable CVs found in the uploaded batch.")

    # Count excluded CVs (grab from any role's full candidate list)
    all_processed = []
    for fname, data in pdf_files.items():
        c = pipeline.process_cv(data, fname)
        if c:
            all_processed.append(c)

    bulk_excluded    = sum(1 for c in all_processed if c.bulk_flag)
    missing_excluded = sum(1 for c in all_processed if c.missing_flag and not c.bulk_flag)
    total_scored     = len(all_processed) - bulk_excluded - missing_excluded

    # Build response
    role_results: Dict[str, RoleResult] = {}
    for role, result in ranking.items():
        candidates_resp = []
        for rank_idx, c in enumerate(result.ranked, start=1):
            candidates_resp.append(CandidateResponse(
                rank            = rank_idx,
                name            = c.name,
                filename        = c.filename,
                score           = round(c.role_scores_norm.get(role, 0.0), 4),
                content_density = round(c.content_density, 2),
                word_count      = c.word_count,
                section_scores  = {k: round(v, 4) for k, v in c.section_scores.get(role, {}).items()},
                top_drivers     = c.top_drivers.get(role, []),
                relevant        = c.relevant.get(role, False),
                ambiguous       = c.ambiguous,
                fl_adjusted     = c.fl_adjusted.get(role, False),
                flags           = build_flags(c, role),
            ))

        role_results[role] = RoleResult(
            role            = role,
            precision_at_5  = round(result.precision_at_5, 3),
            precision_at_10 = round(result.precision_at_10, 3),
            fairness        = build_fairness_response(result.fairness),
            candidates      = candidates_resp,
        )

    return RankResponse(
        total_uploaded   = len(pdf_files),
        total_scored     = total_scored,
        bulk_excluded    = bulk_excluded,
        missing_excluded = missing_excluded,
        results          = role_results,
    )
