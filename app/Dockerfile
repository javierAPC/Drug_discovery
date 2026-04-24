FROM python:3.10-slim

WORKDIR /app

# System dependencies for RDKit
RUN apt-get update && apt-get install -y \
    libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY best_model_part5.pkl .
COPY egfr_bioactivity_cleaned.csv .
COPY X_ecfp6.csv .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
