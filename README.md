# Raspberry Pi Document Cloud

A self-hosted semantic document cloud stack using FastAPI, Celery, Redis, Postgres, and Ollama.

## Setup & Run

1.  **Start the Stack**:
    ```bash
    docker compose up -d --build
    ```

2.  **Pull the LLM Model** (Critical Step):
    Ollama starts empty. You need to pull the `llama3.2` model once.
    ```bash
    docker compose exec ollama ollama pull llama3.2
    ```
    *Note: The metadata for the model is stored in the `ollama_storage` volume, so you only need to do this once.*

3.  **Access the API**:
    The API is available at `http://localhost:8000/docs`.

## Usage

### Upload a Document
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/my_document.pdf" \
  -F "user_id=me"
```
The file will be processed in the background.

### Search Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -F "query_text=Find me the invoice for electric bill"
```

## Directory Structure
- `app/`: Source code.
- `app/uploads`: Mapped to Docker volume `monitor_data`. To use an external HDD on your Pi, update `docker-compose.yml`:
  ```yaml
  volumes:
    monitor_data:
      driver: local
      driver_opts:
        type: none
        o: bind
        device: /mnt/my_external_drive/uploads
  ```
