# Docker for Paddle Inferencing
FROM python:3.8 as base

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
 apt-get install -y \
 python3-pip \
 python-is-python3 \
 ssh \
 git \
 ffmpeg \
 libsm6 \
 libxext6 \
 libpq-dev \
 wget


# install miniconda
FROM base as miniconda-setup

ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash
RUN bash


# Set up SSH:
FROM miniconda-setup as ssh-setup

ARG SSH_PRIVATE_KEY
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_ecdsa
RUN chmod 600 /root/.ssh/id_ecdsa
# Format key correctly
RUN sed -i -e "s/-----BEGIN OPENSSH PRIVATE KEY-----/&\n/"\
    -e "s/-----END OPENSSH PRIVATE KEY-----/\n&/"\
    -e "s/\S\{70\}/&\n/g"\
    /root/.ssh/id_ecdsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
# Ensure ssh is setup correctly:
RUN ssh -T git@github.com ; test $? -eq 1 


# Create conda environment for PaddleDetection using Python 3.8.10
FROM ssh-setup as conda-env-setup

WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml


# PaddleDetection Installation
FROM conda-env-setup as paddle-installation

# GPU version
# RUN conda install -n paddle-venv -y paddlepaddle-gpu==2.4.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
# RUN conda install -n paddle-venv -y paddlepaddle-gpu==2.4.2 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge 
# RUN conda run -n paddle-venv /miniconda3/envs/paddle-venv/bin/pip install paddlepaddle-gpu==2.3.1.post114 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# RUN conda run -n paddle-venv /miniconda3/envs/paddle-venv/bin/pip install paddlepaddle-gpu==2.4.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# https://sarus.readthedocs.io/en/stable/user/custom-cuda-images.html#controlling-the-nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=8.0"

# CPU version
RUN conda install -n paddle-venv -y paddlepaddle==2.4.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/

RUN git clone https://github.com/PaddlePaddle/PaddleDetection.git
WORKDIR /app/PaddleDetection
# Switch to 2.5 release, the one we used to train our model
RUN git checkout release/2.5
# Use the correct pip path
RUN conda run -n paddle-venv /miniconda3/envs/paddle-venv/bin/pip install Cmake numpy 
RUN conda run -n paddle-venv /miniconda3/envs/paddle-venv/bin/pip install -r requirements.txt 
RUN conda run -n paddle-venv /miniconda3/envs/paddle-venv/bin/pip install -e .
RUN conda run -n paddle-venv python setup.py install
# Run tests
# RUN conda run -n paddle-venv python ppdet/modeling/tests/test_architectures.py

# Back to /app directory
WORKDIR /app
# Copy over our configured Paddle files + model params

# Copy over different coco_instance.yml files for each model
COPY ./paddle_files/leaf_instance.yml ./PaddleDetection/configs/datasets/leaf_instance.yml
COPY ./paddle_files/stalk_tiller_braces.yml ./PaddleDetection/configs/datasets/stalk_tiller_braces.yml

# Copy over example labels for each model
COPY ./paddle_files/rogues-leaf-data/labels/test.json ./rogues-leaf-data/labels/test.json
COPY ./paddle_files/rogues-stalk-tiller-braces-data/labels/test.json ./rogues-stalk-tiller-braces-data/labels/test.json

COPY ./paddle_files/solov2_r101_vd_fpn_3x_coco.yml ./PaddleDetection/configs/solov2/solov2_r101_vd_fpn_3x_coco.yml
COPY ./paddle_files/infer.py ./PaddleDetection/tools/infer.py
RUN mkdir ./PaddleDetection/output
RUN mkdir ./PaddleDetection/output/solov2_r101_vd_fpn_3x_coco
# Assumes you have the model params present
COPY ./paddle_files/solo_v2_dec_11_2022_leaf_instance.pdparams ./PaddleDetection/output/solov2_r101_vd_fpn_3x_coco/solo_v2_dec_11_2022_leaf_instance.pdparams
COPY ./paddle_files/solo_v2_sept_05_2023_stalk_tiller_braces.pdparams ./PaddleDetection/output/solov2_r101_vd_fpn_3x_coco/solo_v2_sept_05_2023_stalk_tiller_braces.pdparams


FROM paddle-installation as code-setup

# Copy over our code
COPY ./lambdas ./lambdas

ARG VERSION
ENV VERSION=$VERSION

# Don't buffer Python stdout (logs)
ENV PYTHONUNBUFFERED 1
ENV ECS_AVAILABLE_LOGGING_DRIVERS="awslogs"