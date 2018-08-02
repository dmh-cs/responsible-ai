FROM c12e/cortex-python3-function:latest-master

MAINTAINER CognitiveScale.com

WORKDIR /opt/program
COPY model /opt/program

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "func.py"]