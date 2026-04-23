# Backend

This folder is the inference-facing backend for the project.

Current structure:

- `app/main.py`: FastAPI entrypoint
- `app/routers/inference.py`: upload + predict endpoints
- `app/model_service.py`: model loading and prediction hook
- `app/schemas.py`: response models

The backend is scaffolded so the final chosen X-ray model can be dropped into
`app/model_service.py` without disturbing the research notebooks.

Run locally with:

```bash
uvicorn backend.app.main:app --reload
```

Optional LLM summaries:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5-mini"
export LLM_SUMMARIES_ENABLED="true"
```

When configured, `/predict` returns a short `summary` for each image. The summary
explains the classifier output only; it does not independently diagnose from the
image.

For Render, this repo includes a root [`render.yaml`](/Users/jasonsmith/Desktop/medical-cv-hackathon/render.yaml)
Blueprint that runs the FastAPI app as a single web service. The backend also
serves the static frontend, so one Render service is enough for the demo app.
