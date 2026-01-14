FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV HOME=/home
# Set the working directory
WORKDIR $HOME

# Install system dependencies
RUN apt-get update && apt-get install -y git wget unzip

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN source "$HOME/.cargo/env"

# update pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/utopia-group/TypeT5.git
RUN pip install --no-cache-dir -e TypeT5

COPY . TypeCare 
