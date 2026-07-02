FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY rank.py .

# Small sample bundled for sandbox smoke-testing (see submission_spec.md
# Section 10.5: sandbox only needs to handle a <=100 candidate sample, not
# the full 100K pool).
COPY data/sample_candidates.json ./data/sample_candidates.json

ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "data/sample_candidates.json", "--out", "/app/out/submission.csv", "--top-k", "50"]
