"""
Basic smoke tests for the RAWSM pipeline (no real PDFs needed).
Run with: pytest tests/
"""

import io
import pytest
from unittest.mock import patch, MagicMock

from app.pipeline import (
    preprocess,
    detect_sections,
    content_density,
    length_tier,
    top_section_drivers,
    TECH_NORMALIZE,
)
from app.config import JOB_DESCRIPTIONS, ROLE_WEIGHTS


# ── preprocess ────────────────────────────────────────────────────────────────

def test_preprocess_removes_email():
    result = preprocess("Contact me at john.doe@example.com for details.")
    assert "@" not in result

def test_preprocess_removes_url():
    result = preprocess("Visit http://example.com for more info.")
    assert "http" not in result

def test_preprocess_expands_nlp():
    result = preprocess("Experienced in NLP and ML.")
    assert "natural language processing" in result
    assert "machine learning" in result

def test_preprocess_empty_input():
    assert preprocess("") == ""
    assert preprocess("   ") == ""

def test_preprocess_short_input():
    assert preprocess("hi") == ""


# ── detect_sections ───────────────────────────────────────────────────────────

def test_detect_sections_finds_skills():
    text = "Skills\npython pytorch tensorflow\nEducation\nBSc Computer Science"
    sections = detect_sections(text)
    assert "python" in sections["skills"].lower()
    assert "bsc" in sections["education"].lower()

def test_detect_sections_defaults_to_general():
    text = "Some random text without headers"
    sections = detect_sections(text)
    assert len(sections["general"]) > 0

def test_detect_sections_all_keys_present():
    sections = detect_sections("")
    expected = {"summary", "experience", "projects", "skills", "education", "certifications", "general"}
    assert expected == set(sections.keys())


# ── content_density ───────────────────────────────────────────────────────────

def test_content_density_zero_words():
    assert content_density({}, 0) == 0.0

def test_content_density_full():
    sections = {"experience": "a b c d e", "general": ""}
    assert content_density(sections, 5) == 1.0

def test_content_density_partial():
    sections = {"experience": "a b", "general": "c d e f"}
    density = content_density(sections, 6)
    assert 0.0 < density < 1.0


# ── length_tier ───────────────────────────────────────────────────────────────

def test_length_tier_short():
    assert length_tier(100) == 0

def test_length_tier_medium():
    assert length_tier(350) == 1

def test_length_tier_long():
    assert length_tier(600) == 2

def test_length_tier_boundaries():
    assert length_tier(199) == 0
    assert length_tier(200) == 1
    assert length_tier(500) == 1
    assert length_tier(501) == 2


# ── top_section_drivers ───────────────────────────────────────────────────────

def test_top_section_drivers_returns_top_n():
    scores = {"experience": 0.9, "skills": 0.7, "projects": 0.5, "general": 0.1}
    drivers = top_section_drivers(scores, n=2)
    assert drivers == ["experience", "skills"]

def test_top_section_drivers_excludes_zeros():
    scores = {"experience": 0.8, "skills": 0.0, "projects": 0.0}
    drivers = top_section_drivers(scores)
    assert "skills" not in drivers
    assert "projects" not in drivers


# ── config validation ─────────────────────────────────────────────────────────

def test_role_weights_sum_to_one():
    for role, weights in ROLE_WEIGHTS.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"{role} weights sum to {total}, expected 1.0"

def test_all_roles_have_job_descriptions():
    assert set(ROLE_WEIGHTS.keys()) == set(JOB_DESCRIPTIONS.keys())

def test_job_descriptions_not_empty():
    for role, jd in JOB_DESCRIPTIONS.items():
        assert len(jd.strip()) > 50, f"JD for {role} is too short"
