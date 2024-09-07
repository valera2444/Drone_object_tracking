FROM python:3.11
WORKDIR /usr/src/app
COPY deep_sort_realtime ./deep_sort_realtime
COPY example_for_docker_image ./example_for_docker_image
COPY torchreid ./torchreid
COPY sort.py utils.py run.py ./
COPY YOLOv8x3n.pt ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 5000
ENTRYPOINT ["python3.11", "run.py"]
