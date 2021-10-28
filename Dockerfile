FROM jinaai/jina:2-py37-perf

# install requirements before copying the workspace
RUN apt-get -y update && apt-get install -y git build-essential

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]