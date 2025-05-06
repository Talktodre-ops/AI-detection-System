   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   # Download NLTK data if needed
   RUN python -c "import nltk; nltk.download('punkt')"

   # Make sure the model directory exists
   RUN mkdir -p src/model/models/distilroberta

   CMD ["streamlit", "run", "src/frontend/mixed_content_app.py", "--server.port=7860", "--server.address=0.0.0.0"]