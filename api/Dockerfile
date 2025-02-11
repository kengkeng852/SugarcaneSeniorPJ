# Use the official Python 3.10.11 slim image
FROM python:3.10.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . .

# Install dependencies from requirements.txt without using cache
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (Cloud Run expects your container to listen on this port)
EXPOSE 8080

# Start the FastAPI application using Uvicorn
# Use shell form so that environment variable expansion works
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
