# role-aware-weighted-semantic-matching

Fairness-aware CV screening API. Ranks candidates against role-specific job descriptions using section-level semantic similarity, content-density scoring, and FairLearn bias calibration.

## Architecture

```
app/
├── main.py       # FastAPI routes and response schemas
├── pipeline.py   # Core RAWSM logic (extraction, scoring, fairlearn)
└── config.py     # Job descriptions and role weights
tests/
└── test_pipeline.py
Dockerfile
docker-compose.yml
requirements.txt
```

## Quickstart

```bash
# Build and run
docker compose up --build

# API is live at
http://localhost:8000

# Interactive docs
http://localhost:8000/docs
```

## Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| `GET`  | `/health` | Liveness check                       |
| `GET`  | `/roles`  | List configured roles + weights      |
| `POST` | `/rank`   | Upload CVs, receive ranked results   |

### POST /rank

Upload one or more PDF CVs as `multipart/form-data`:

```bash
curl -X POST http://localhost:8000/rank \
  -F "files=@cv1.pdf" \
  -F "files=@cv2.pdf" \
  -F "files=@cv3.pdf"
```

**Response structure:**

```json
{
  "total_uploaded": 3,
  "total_scored": 3,
  "bulk_excluded": 0,
  "missing_excluded": 0,
  "results": {
    "AI Engineer": {
      "role": "AI Engineer",
      "precision_at_5": 0.8,
      "precision_at_10": 0.7,
      "fairness": {
        "dpd_before": 0.127,
        "dpd_after": 0.052,
        "n_adjusted": 5,
        "tier_rates": { "Medium (200–500w)": 0.5, "Long (>500w)": 0.552 },
        "note": "DPD reduced by 59.1% after adjustment."
      },
      "candidates": [
        {
          "rank": 1,
          "name": "Jane Smith",
          "filename": "jane_smith.pdf",
          "score": 0.9821,
          "content_density": 0.74,
          "word_count": 620,
          "section_scores": { "experience": 0.812, "skills": 0.743, ... },
          "top_drivers": ["experience", "skills", "projects"],
          "relevant": true,
          "ambiguous": false,
          "fl_adjusted": false,
          "flags": []
        }
      ]
    }
  }
}
```

## Adding a new role

Edit `app/config.py`:

```python
JOB_DESCRIPTIONS["Data Engineer"] = """..."""

ROLE_WEIGHTS["Data Engineer"] = {
    "summary"       : 0.10,
    "experience"    : 0.35,
    "projects"      : 0.20,
    "skills"        : 0.25,
    "education"     : 0.05,
    "certifications": 0.03,
    "general"       : 0.02,
}
```

Weights must sum to 1.0. Rebuild the container.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## Design decisions

- **Section-level scoring** — each CV section is embedded and scored independently against the JD, then combined via role-specific weights. This outperforms whole-document similarity for structured documents.
- **Content-density penalty** — CVs with poor section structure lose up to 15% of their raw score, penalising unstructured dumps without hard-excluding them.
- **FairLearn ThresholdOptimizer** — post-processing bias mitigation using CV length tiers (Short / Medium / Long) as the sensitive feature. Short CVs are merged into Medium when their group is too small for calibration.
- **Bulk PDF exclusion** — files exceeding 5 pages or 3,000 words are flagged and excluded from scoring to prevent merged/combined PDFs from dominating the ranking.
