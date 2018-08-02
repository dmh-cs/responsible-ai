FROM smizy/scikit-learn

MAINTAINER CognitiveScale.com

WORKDIR /opt/program
COPY model /opt/program

RUN pip install cortex-client

ENTRYPOINT ["python", "func.py"]