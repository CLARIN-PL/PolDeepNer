#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

#
# Download kgr10.plain.skipgram.dim300.neg10.bin from CLARIN-PL NextCloud
#
EMBEDDING="./poldeepner/model/kgr10.plain.skipgram.dim300.neg10.bin"
EMBEDDING_URL="https://nextcloud.clarin-pl.eu/index.php/s/luubhnS0AvjmtQc/download?path=%2F&files=kgr10.plain.skipgram.dim300.neg10.bin"

if [ ! -f $EMBEDDING ]; then
  wget $EMBEDDING_URL -O $EMBEDDING
  echo "- `basename $EMBEDDING` downloaded"
else
  echo "- `basename $EMBEDDING` found"
fi


#
# Download KGR10 model from CLARIN-PL NextCloud
#
function download_nextcloud()
{
    MODEL=$1
    MODEL_PACK="$MODEL.7z"
    MODEL_URL=$2

    if [ ! -f $MODEL_PACK ] && [ ! -f $MODEL ]; then
      wget $MODEL_URL -O $MODEL_PACK
      echo "- `basename $MODEL_PACK` downloaded"
    else
      echo "- `basename $MODEL_PACK` found"
    fi

    if [ ! -f $MODEL ]; then
      OUTPUT=`dirname $MODEL`
      7z x $MODEL_PACK -o $OUTPUT
      echo "- `basename $MODEL_PACK` unpacked"
    else
      echo "- `basename $MODEL` found"
    fi
}

#
# Unpack pre-trained models
#
function unpack_model_7z()
{
  MODEL=$1
  if [ ! -d $MODEL ] ; then
    7z e "$MODEL.7z" -o$MODEL
    echo "- `basename $MODEL` unpacked"
  else
    echo "- `basename $MODEL` found"
  fi
}

unpack_model_7z "./poldeepner/model/poldeepner-kgr10.plain.skipgram.dim300.neg10.bin"
