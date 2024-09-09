# Stage 1: Build Stage
FROM python:3.11 AS builder

# Set working directory
WORKDIR /usr/src/app

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt .

# Install dependencies without cache to keep the layer size minimal
RUN pip install --no-cache-dir -r requirements.txt

COPY deep_sort_realtime ./deep_sort_realtime
COPY example_for_docker_image ./example_for_docker_image
COPY torchreid ./torchreid
COPY sort.py utils.py run.py ./
COPY YOLOv8x3n.pt ./

# Stage 2: Final Image Stage
FROM python:3.11-slim

# Set working directory
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --from=builder /usr/src/app/deep_sort_realtime ./deep_sort_realtime
COPY --from=builder /usr/src/app/example_for_docker_image ./example_for_docker_image
COPY --from=builder /usr/src/app/torchreid ./torchreid
COPY --from=builder /usr/src/app/sort.py /usr/src/app/utils.py /usr/src/app/run.py ./
COPY --from=builder /usr/src/app/YOLOv8x3n.pt ./

# Expose port
EXPOSE 5000

# Set entrypoint
ENTRYPOINT ["python3.11", "run.py"]