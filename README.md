# Cataneration Trophy Room

Streamlit app for the Cataneration fantasy football history dataset (2013-present).

## Local run

```bash
pip install -r requirements.txt
streamlit run src/ui/app.py
```

## Streamlit Community Cloud deployment

1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app from the repo.
3. Set the main file to `src/ui/app.py` and deploy.

The app reads prebuilt CSVs from `data/mart`.
