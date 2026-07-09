FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (smaller layer, better caching)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY demo/requirements.txt demo-requirements.txt
RUN pip install --no-cache-dir -r demo-requirements.txt

COPY src/ src/
COPY demo/ demo/

EXPOSE 7860

CMD ["python", "demo/app.py"]