"""
RAWSM — Job descriptions and role-specific section weights.
Edit this file to add or modify roles without touching pipeline logic.
"""

JOB_DESCRIPTIONS = {
    "AI Engineer": """
We are seeking an AI Engineer to design and develop intelligent systems
for candidate assessment and job matching using multimodal data.
The role involves building AI models for resume understanding using NLP,
analyzing interviews through text, audio, and video modalities,
and designing candidate ranking and scoring systems.
The engineer will implement LLM-based pipelines including RAG and prompt engineering,
apply fairness-aware and bias mitigation techniques,
and build explainable AI components for decision transparency using tools like LIME and SHAP.
Strong Python programming is required along with deep understanding of
machine learning, deep learning, transformers, embeddings, and semantic similarity.
Experience with PyTorch or TensorFlow, Hugging Face frameworks, and model deployment is expected.
The ideal candidate understands model evaluation metrics and can optimize
systems for performance and scalability.
""",
    "Full Stack Engineer": """
We are seeking a Full Stack Engineer to develop scalable systems
for an AI-powered recruitment platform.
Responsibilities include developing and maintaining frontend interfaces using React,
building backend APIs with Node.js, Django, or FastAPI,
and integrating AI models into production systems.
The engineer will design and manage both structured databases like PostgreSQL
and unstructured databases like MongoDB and vector databases.
Building scalable pipelines for CV processing and candidate data management is required,
along with ensuring system performance, security, and reliability.
Strong experience with JavaScript, TypeScript, REST APIs, and database design is required.
Familiarity with Docker, cloud deployment on AWS or Azure,
vector search systems, and real-time scalable pipelines is preferred.
Understanding of software architecture and system design principles is essential.
""",
}

# Section weights must sum to 1.0 per role.
ROLE_WEIGHTS = {
    "AI Engineer": {
        "summary"       : 0.10,
        "experience"    : 0.30,
        "projects"      : 0.25,
        "skills"        : 0.20,
        "education"     : 0.05,
        "certifications": 0.05,
        "general"       : 0.05,
    },
    "Full Stack Engineer": {
        "summary"       : 0.10,
        "experience"    : 0.30,
        "projects"      : 0.25,
        "skills"        : 0.20,
        "education"     : 0.05,
        "certifications": 0.05,
        "general"       : 0.05,
    },
}
