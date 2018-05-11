#import nvidia-docker image with latest cudnn
FROM nvidia/cuda:9.1-cudnn7-devel

RUN apt-get update
RUN apt-get install -y build-essential git libopenblas-dev liblapack-dev libopencv-dev python-pip python-dev python-setuptools python-numpy
RUN git clone --single-branch --depth 1 --branch v0.11.0 --recursive https://github.com/apache/incubator-mxnet
RUN nice -n20 make -C incubator-mxnet -j`nproc` USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_BLAS=openblas
RUN pip2 install -e incubator-mxnet/python

RUN apt-get install -y python3 python3-pip
RUN pip3 install numpy scikit-image
RUN mkdir -p fashion-mnist
RUN apt-get install -y wget
RUN wget https://raw.githubusercontent.com/illinois-impact/ece408_project/master/reader.py
RUN wget -P fashion-mnist \
        https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/scripts/generate-data.py \
        https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/scripts/reader.py
RUN chmod +x fashion-mnist/generate-data.py
RUN fashion-mnist/generate-data.py fashion-mnist
RUN mkdir -p models
RUN wget -O models/baseline-0002.params https://github.com/illinois-impact/ece408_mxnet_docker/blob/2018sp/models/baseline-0002.params?raw=true
RUN wget -O models/baseline-symbol.json https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/models/baseline-symbol.json?raw=true
RUN wget -O models/ece408-0002.params https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/models/ece408-0002.params?raw=true
RUN wget -O models/ece408-symbol.json https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/models/ece408-symbol.json?raw=true
RUN wget -O m1.1.py https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/scripts/m1.1.py?raw=true
RUN wget -O m1.2.py https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/scripts/m1.2.py?raw=true
RUN wget -O m2.1.py https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/scripts/m2.1.py?raw=true
RUN wget -O m3.1.py https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/scripts/m3.1.py?raw=true
RUN wget -O m4.1.py https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/scripts/m4.1.py?raw=true

RUN wget -O incubator-mxnet/src/operator/custom/new.cc https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/ece408_src/new.cc?raw=true
RUN wget -O incubator-mxnet/src/operator/custom/new.cu https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/ece408_src/new.cu?raw=true
RUN wget -O incubator-mxnet/src/operator/custom/new-inl.h https://github.com/illinois-impact/ece408_mxnet_docker/raw/2018sp/ece408_src/new-inl.h?raw=true

#COPY ece408_src/* incubator-mxnet/src/operator/custom/
#RUN make -C incubator-mxnet USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
