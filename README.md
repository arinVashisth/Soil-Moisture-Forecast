# Soil Moisture Forecast (Flask)

Short description: Flask app that predicts soil moisture using pre-trained models.
How to run locally:
1. `pip install -r requirements.txt`
2. `python app.py` (or `gunicorn app:app`)

Contents:
- app.py
- templates/
- static/
- models/  (put small models here or use cloud storage)
- Data - *.csv

Notes: If your model files are large (>50 MB) consider using Git LFS or storing them in S3.
