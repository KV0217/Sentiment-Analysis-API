# Twitter Sentiment Analysis API

Production NLP API — 76.9% accuracy (133% lift over baseline) with ABSA department routing.

## Live
| | URL |
|--|--|
| API | https://sentiment-analysis-api-oegw.onrender.com |
| Docs | https://sentiment-analysis-api-oegw.onrender.com/docs |
| Streamlit | https://sentiment-analysis-api-kv.streamlit.app |

## Screenshots
![API Docs](screenshots/api_docs.png)
![Single Tweet Response](screenshots/api_response.png)
![Batch Response](screenshots/batch_response.png)

## Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /predict | Single tweet sentiment + ABSA |
| POST | /predict/batch | Batch tweets + department summary |

## What Makes This Different
- ABSA automatically routes complaints to the right department
- Batch endpoint returns department-level complaint counts
- 99.7% confidence on clear negative tweets

## Run Locally
```bash
git clone https://github.com/KV0217/Sentiment-Analysis-API.git
cd Sentiment-Analysis-API
pip install -r requirements.txt
uvicorn main:app --reload
```

## Tech Stack
FastAPI · TF-IDF · Scikit-learn · Docker · Render

## Related
- Analysis notebook: [Sentiment-Analysis](https://github.com/KV0217/sentiment-analysis)
