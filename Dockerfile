#---------------------------------------------------------------
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 as poldeepner-base
#---------------------------------------------------------------

LABEL maintainer="Michał Marcińczuk <marcinczuk@gmail.com>"

RUN apt-get clean && apt-get update

# Set the locale
RUN apt-get install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# update pip
RUN apt-get install -y git
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel


#---------------------------------------------------------------
FROM poldeepner-base
#---------------------------------------------------------------

RUN pip install seqeval
RUN pip install keras
RUN pip install tensorflow
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN pip install cython
RUN pip install pyfasttext
RUN pip install fasttext
RUN pip install sklearn
RUN pip install scikit-learn==0.22.2.post1
RUN pip install python-dateutil
RUN pip install nltk
RUN pip install gensim
RUN pip install allennlp==0.9.0

WORKDIR "/poldeepner"
