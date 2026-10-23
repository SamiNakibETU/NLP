# Image GPU : API + interface Gradio sur le port 7860.
#   docker build -t porteparole .
#   docker run --gpus all -p 7860:7860 -v /chemin/hf_cache:/cache -v /chemin/NLP:/nlp:ro \
#       -e HF_TOKEN=... -e PORTEPAROLE_NLP_REPO=/nlp porteparole
# Les poids ne sont pas dans l'image : ils se téléchargent dans /cache au premier appel de chaque moteur.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt
COPY . .

ENV HF_HOME=/cache PYTHONIOENCODING=utf-8 PORTEPAROLE_MT=off PORTEPAROLE_NLP_REPO=/nlp
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:7860/api/health')"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
