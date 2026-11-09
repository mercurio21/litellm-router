FROM ghcr.io/berriai/litellm-database:main-stable

COPY config.yaml /app/config.yaml
COPY anthropic_oauth_identity.py /app/anthropic_oauth_identity.py

CMD ["--config", "/app/config.yaml", "--port", "4000"]
