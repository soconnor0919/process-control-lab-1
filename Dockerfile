# Use a slim version of Python to keep the image size down
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files 
# and to ensure output is sent straight to terminal (useful for logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for Matplotlib and OpenPyXL
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install specific Python requirements
# We include scipy because first-order fitting usually relies on curve_fit or least_squares
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    matplotlib \
    openpyxl \
    scipy

# Copy the rest of the application code
COPY . .

# Streamlit runs on 8501 by default
EXPOSE 8501

# Run the application
# --server.address=0.0.0.0 is mandatory for Docker networking
# --server.headless=true prevents Streamlit from trying to open a browser window
ENTRYPOINT ["streamlit", "run", "gui_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
