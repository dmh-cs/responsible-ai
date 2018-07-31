FROM pytorch/pytorch:latest

MAINTAINER CognitiveScale.com

WORKDIR /opt/program
COPY model /opt/program

RUN pip install cortex-client

ENTRYPOINT ["python", "func.py"]

# python run.py --model-path ./resnet18-5c106cde.pth --label-path labels.json --img-path "https://images.pexels.com/photos/50704/car-race-ferrari-racing-car-pirelli-50704.jpeg"