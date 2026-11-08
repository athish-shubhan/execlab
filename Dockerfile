# Build from the directory that contains both the skeptic and execlab checkouts:
#   docker build -f execlab/Dockerfile -t execlab .
FROM python:3.11-slim
WORKDIR /work
COPY skeptic /work/skeptic
COPY execlab /work/execlab
RUN pip install --no-cache-dir -e /work/skeptic -e "/work/execlab[recorder,dev]"
WORKDIR /work/execlab
CMD ["make", "full"]
