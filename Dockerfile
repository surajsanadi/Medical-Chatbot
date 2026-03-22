FROM python:3.11-slim

WORKDIR /app

# copy dependency file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy application
COPY . .

# download nltk resources
RUN python -m nltk.downloader punkt punkt_tab

# container port
EXPOSE 8000

# run production server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "600", "app:app"]