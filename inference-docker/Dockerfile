FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y git \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :


# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH


# Install ASAP
RUN : \
    && apt-get update \
    && apt-get -y install curl \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.1-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.1-Ubuntu2004.deb \
    && :


# Install OpenSlide dependencies
RUN : \
    && apt-get update \
    && apt-get install -y openslide-tools libopenslide0 \
    && apt-get install -y build-essential libffi-dev libxml2-dev libjpeg-turbo8-dev zlib1g-dev \
    && apt-get clean \
    && :

# Install OpenSlide Python bindings
RUN /venv/bin/python3.8 -m pip install --no-cache-dir openslide-python

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# update permissions
RUN chown -R user:user /venv/

USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

# Update pip
RUN /venv/bin/python3.8 -m pip install pip --upgrade


# You can add any Python dependencies to requirements.txt
RUN /venv/bin/python3.8 -m pip install \
    --no-cache-dir \
    -r /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user structures.py /opt/app/
COPY --chown=user:user wsdetectron2.py /opt/app/


#install pytorch
RUN /venv/bin/python3.8 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Verify torch installation to ensure it's available
RUN /venv/bin/python3.8 -c "import torch; print(torch.__version__)"

## Install detectron2
RUN /venv/bin/python3.8 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
# Install Whole Slide Data
RUN /venv/bin/python3.8 -m pip install 'git+https://github.com/DIAGNijmegen/pathology-whole-slide-data@main'

USER user
ENTRYPOINT ["/venv/bin/python3.8", "inference.py"]
