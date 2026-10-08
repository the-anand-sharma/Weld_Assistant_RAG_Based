# Container for the Weld Quality Agent (ADK web UI).
# Built for Hugging Face Spaces, which runs the container as user id 1000
# and expects the app on port 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install dependencies first so Docker can cache this layer between builds.
COPY --chown=user weld_agent/requirements.txt weld_agent/requirements.txt
RUN pip install --no-cache-dir -r weld_agent/requirements.txt

# Download the embedding model at build time, so the first question after a
# restart does not have to wait for it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user weld_agent weld_agent

EXPOSE 7860
# GOOGLE_API_KEY is not in the image: it is injected at runtime as a secret.
CMD ["adk", "web", "--host", "0.0.0.0", "--port", "7860", "--session_service_uri", "memory://"]
