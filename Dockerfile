# ------------------------------
# 1️⃣ Base image
# ------------------------------
FROM python:3.10-slim

# ------------------------------
# 2️⃣ Set working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# 3️⃣ Copy everything into container
# ------------------------------
COPY . /app

# ------------------------------
# 4️⃣ System dependencies & pip upgrade
# ------------------------------
RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip

# ------------------------------
# 5️⃣ Install Python dependencies
# ------------------------------
RUN pip install --default-timeout=100 --retries 5 --no-cache-dir -r requirements.txt

# ------------------------------
# 6️⃣ Expose FastAPI port
# ------------------------------
EXPOSE 8000

# ------------------------------
# 7️⃣ Start API
# ------------------------------
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]