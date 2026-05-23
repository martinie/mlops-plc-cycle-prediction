FROM python:3.11-slim

# Slim is being used because it's lightweight and more secure
# I know I lost marks in the last assignment for using but I find
# that using it makes better sense than the full python suite

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY data/raw/ data/raw/

RUN python -m src.preprocess
RUN python -m src.train

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.main:app"]