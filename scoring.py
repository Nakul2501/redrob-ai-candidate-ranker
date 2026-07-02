"""
jd_profile.py
==============
A structured, hand-curated encoding of the "Senior AI Engineer — Founding
Team" job description at Redrob AI.

WHY THIS FILE EXISTS
---------------------
The JD explicitly warns that the "right answer" is NOT "find candidates
whose skills section contains the most AI keywords." Naive keyword counting
(see sample_submission.csv, which ranks an HR Manager #1 for having "9 AI
core skills") is the trap the hackathon is designed to catch.

Instead of a black-box model, this file is the *transparent, auditable*
encoding of what the JD actually says. Every weight and keyword list below
traces back to a specific sentence in job_description.docx (cited in
comments). If a reviewer wants to know "why did candidate X rank where they
did," the answer is: "these concept clusters matched with this much trust,
these disqualifiers did/didn't fire, here is the arithmetic" — not "the
model decided."

This also *is* the feature-importance / feature-selection artifact the
hackathon asks for: the component weights in scoring.py + the keyword
clusters here together constitute the full, inspectable feature set.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. CORE MUST-HAVE CONCEPT CLUSTERS
#    JD: "Things you absolutely need"
#    Each cluster is a bag of terms/phrases that, if found in a candidate's
#    skills, summary, or career_history descriptions, count as evidence of
#    that concept. This is deliberately phrase-level and synonym-aware so
#    that a candidate who never writes the word "retrieval" but writes
#    "built a semantic search system over 10M documents" still matches the
#    RETRIEVAL cluster. This is what lets a plain-language Tier-5 candidate
#    (JD's own example) be recognized without keyword-stuffing.
# ---------------------------------------------------------------------------

CORE_CLUSTERS: dict[str, list[str]] = {
    "embeddings_retrieval": [
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "openai embeddings", "bge", "e5 embedding", "dense retrieval",
        "semantic search", "vector search", "nearest neighbor search", "ann search",
        "approximate nearest neighbor", "similarity search", "text embedding",
        "representation learning", "retrieval augmented", "rag", "dual encoder",
        "bi-encoder", "cross-encoder", "embedding drift", "index refresh",
    ],
    "vector_db_hybrid_search": [
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
        "faiss", "vector database", "vector db", "hybrid search", "hybrid retrieval",
        "bm25", "lucene", "solr", "inverted index", "ivf index", "hnsw",
        "vespa", "typesense", "chroma", "chromadb", "pgvector",
    ],
    "python": [
        "python", "pandas", "numpy", "pytorch", "tensorflow", "fastapi", "django",
        "flask", "scikit-learn", "sklearn",
    ],
    "eval_frameworks": [
        "ndcg", "mrr", "mean reciprocal rank", "map metric", "mean average precision",
        "offline evaluation", "online evaluation", "a/b test", "ab test",
        "offline-to-online correlation", "recruiter-feedback loop", "recruiter feedback loop",
        "click-through", "ctr", "relevance evaluation", "eval harness", "eval framework",
        "golden set", "labeled relevance", "precision@", "recall@", "hit rate",
    ],
}

# JD: "Things we'd like you to have but won't reject you for"
NICE_TO_HAVE_CLUSTERS: dict[str, list[str]] = {
    "llm_finetuning": [
        "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetune",
        "instruction tuning", "rlhf", "dpo", "supervised fine-tuning", "sft",
    ],
    "learning_to_rank": [
        "learning to rank", "learning-to-rank", "ltr", "xgboost ranker",
        "lambdamart", "lambdarank", "neural ranking", "pairwise ranking",
        "listwise ranking", "pointwise ranking", "reranker", "re-ranking", "reranking",
    ],
    "hr_recruiting_marketplace": [
        "recruiting", "recruiter", "ats", "applicant tracking", "job matching",
        "candidate matching", "marketplace", "two-sided marketplace", "hr-tech", "hr tech",
        "talent", "hiring platform",
    ],
    "distributed_scale": [
        "distributed systems", "large-scale inference", "high throughput",
        "low latency serving", "model serving", "spark", "kafka", "kubernetes",
        "sharding", "horizontal scaling", "load balancing", "quantization",
        "batching", "onnx", "triton inference",
    ],
    "open_source": [
        "open source", "open-source", "github stars", "published paper", "arxiv",
        "conference talk", "meetup talk", "oss contributor", "maintainer",
    ],
}

# JD: "The right answer involves reasoning about the gap between what the JD
# says and what the JD means" -- these are broader signals that a candidate
# has genuinely SHIPPED a ranking/search/recsys system, as opposed to just
# having studied one. Used to separate keyword-holders from doers, and to
# recognize plain-language Tier-5s.
SHIPPED_SYSTEM_SIGNALS: list[str] = [
    "recommendation system", "recommender system", "search ranking", "search relevance",
    "ranking system", "matching system", "personalization", "feed ranking",
    "query understanding", "candidate generation", "shipped to production",
    "in production", "real users", "at scale", "production system", "deployed to users",
    "serving live traffic", "millions of users", "requests per second", "qps",
]

# Signals of pure-research / non-production work (used to detect the
# "pure research environments... without production deployment" disqualifier)
RESEARCH_ONLY_SIGNALS: list[str] = [
    "phd thesis", "doctoral research", "academic lab", "research lab",
    "published paper", "arxiv preprint", "postdoc", "research intern",
    "research assistant", "university lab", "no production", "benchmark only",
]

# ---------------------------------------------------------------------------
# 2. TITLE / SENIORITY LANGUAGE
# ---------------------------------------------------------------------------

# JD: role is IC-heavy, "writes code." Titles suggesting hands-on AI/ML/search
# engineering, ordered loosely by relevance (not seniority).
RELEVANT_TITLE_KEYWORDS: list[str] = [
    "machine learning", "ml engineer", "ai engineer", "applied scientist",
    "research engineer", "search engineer", "search ", "retrieval engineer",
    "recommendation", "ranking engineer", "nlp engineer", "data scientist",
    "mle", "ai/ml", "recsys",
]

# JD: "senior engineer who hasn't written production code in the last 18
# months because you've moved into 'architecture' or 'tech lead' roles" is a
# soft disqualifier for the CURRENT role only.
ARCHITECTURE_ONLY_TITLE_KEYWORDS: list[str] = [
    "architect", "engineering manager", "director of engineering", "vp engineering",
    "head of engineering", "principal architect", "chief architect",
]

# JD: "title-chasers... optimizing for Senior -> Staff -> Principal by
# switching companies every 1.5 years"
TITLE_LEVEL_WORDS: list[str] = ["junior", "associate", "senior", "staff", "principal", "lead", "chief"]

# ---------------------------------------------------------------------------
# 3. EXPLICIT DISQUALIFIERS ("Things we explicitly do NOT want")
# ---------------------------------------------------------------------------

# JD: "People who have only worked at consulting firms (TCS, Infosys, Wipro,
# Accenture, Cognizant, Capgemini, etc.) in their entire career."
KNOWN_CONSULTING_FIRMS: set[str] = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcltech", "tech mahindra", "mindtree",
    "l&t infotech", "ltimindtree", "mphasis", "persistent systems", "birlasoft",
    "hexaware", "zensar",
}
CONSULTING_INDUSTRY_LABELS: set[str] = {"it services", "consulting", "it consulting"}

# JD: "People whose primary expertise is computer vision, speech, or
# robotics without significant NLP/IR exposure"
CV_SPEECH_ROBOTICS_CLUSTER: list[str] = [
    "computer vision", "image classification", "object detection", "image segmentation",
    "gan", "gans", "generative adversarial", "speech recognition", "asr",
    "text-to-speech", "tts", "robotics", "slam", "autonomous navigation",
    "lidar", "motion planning", "reinforcement learning for robotics",
]
NLP_IR_CLUSTER: list[str] = [
    "nlp", "natural language processing", "information retrieval", "text classification",
    "named entity recognition", "ner", "search", "retrieval", "language model",
    "llm", "text mining", "topic modeling", "sentiment analysis", "tokenization",
    "question answering", "summarization",
]

# JD: framework-enthusiast pattern -- lots of trendy-framework surface, no
# systems depth (no vector DB / infra / eval keywords alongside it).
FRAMEWORK_SURFACE_CLUSTER: list[str] = [
    "langchain", "llamaindex", "autogpt", "crewai", "langgraph", "haystack",
    "semantic kernel", "flowise", "n8n",
]

# ---------------------------------------------------------------------------
# 4. LOCATION / LOGISTICS
# ---------------------------------------------------------------------------

PREFERRED_CITIES: set[str] = {"noida", "pune"}
EXPLICIT_WELCOME_CITIES: set[str] = {"hyderabad", "pune", "mumbai", "delhi", "new delhi",
                                       "noida", "gurgaon", "gurugram"}
# Other well-known Indian tech hubs -- JD doesn't exclude these, "hybrid,
# flexible cadence," so treat generously rather than penalize by omission.
OTHER_INDIA_TECH_HUBS: set[str] = {
    "bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad", "kochi",
    "trivandrum", "coimbatore", "indore", "chandigarh", "vizag", "bhubaneswar",
}

# ---------------------------------------------------------------------------
# 5. EXPERIENCE BAND
#    JD: "5-9 years... a range, not a requirement." Ideal candidate: "6-8
#    years total... 4-5 in applied ML/AI roles at product companies."
# ---------------------------------------------------------------------------

EXPERIENCE_IDEAL_LOW = 6.0
EXPERIENCE_IDEAL_HIGH = 8.0
EXPERIENCE_BAND_LOW = 5.0
EXPERIENCE_BAND_HIGH = 9.0

# ---------------------------------------------------------------------------
# 6. NOTICE PERIOD
#    JD: "sub-30-day notice... can buy out up to 30 days... 30+ day
#    candidates still in scope but the bar gets higher."
# ---------------------------------------------------------------------------

NOTICE_FULL_CREDIT_DAYS = 30
NOTICE_ZERO_CREDIT_DAYS = 120  # beyond this, treat as heavily discounted, not zero
