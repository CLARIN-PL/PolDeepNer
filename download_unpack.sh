#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

#
# Download and unpack cc.pl.300.bin
#
FASTTEXT_CC="$DIR/poldeepner/model/cc.pl.300.bin"
FASTTEXT_CC_PACK="$DIR/poldeepner/model/cc.pl.300.bin.gz"
FASTTEXT_CC_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz"

if [ ! -f $FASTTEXT_CC_PACK ] && [ ! -f $FASTTEXT_CC ]; then
  wget $FASTTEXT_CC_URL -O $FASTTEXT_CC_PACK
  echo "- `basename $FASTTEXT_CC_PACK` downloaded"
else
  echo "- `basename $FASTTEXT_CC_PACK` found"
fi

if [ ! -f $FASTTEXT_CC ]; then
  gunzip $FASTTEXT_CC_PACK
  echo "- `basename $FASTTEXT_CC_PACK` unpacked"
else
  echo "- `basename $FASTTEXT_CC` found"
fi


#
# Download KGR10 embeddings from CLARIN-PL NextCloud
#
function download_nextcloud()
{
    FASTTEXT_KGR10=$1
    FASTTEXT_KGR10_PACK="$FASTTEXT_KGR10.7z"
    FASTTEXT_KGR10_URL=$2

    if [ ! -f $FASTTEXT_KGR10_PACK ] && [ ! -f $FASTTEXT_KGR10 ]; then
      wget $FASTTEXT_KGR10_URL -O $FASTTEXT_KGR10_PACK
      echo "- `basename $FASTTEXT_KGR10_PACK` downloaded"
    else
      echo "- `basename $FASTTEXT_KGR10_PACK` found"
    fi

    if [ ! -f $FASTTEXT_KGR10 ]; then
      OUTPUT=`dirname $FASTTEXT_KGR10`
      7z x $FASTTEXT_KGR10_PACK -o$OUTPUT
      echo "- `basename $FASTTEXT_KGR10_PACK` unpacked"
    else
      echo "- `basename $FASTTEXT_KGR10` found"
    fi
}

#
# Download KGR10 embeddings from CLARIN-PL NextCloud
#
function download_nextcloud_raw()
{
    FILE_LOCAL=$1
    FILE_URL=$2

    if [ ! -f FILE_LOCAL ]; then
      wget $FILE_URL -O $FILE_LOCAL
      echo "- `basename FILE_LOCAL` downloaded"
    else
      echo "- `basename FILE_LOCAL` found"
    fi
}

download_nextcloud "$DIR/poldeepner/model/kgr10-plain-sg-300-mC50.bin" "https://nextcloud.clarin-pl.eu/index.php/s/HIFaRv7ekgw24F1/download"
download_nextcloud "$DIR/poldeepner/model/kgr10_orths.vec.bin" "https://nextcloud.clarin-pl.eu/index.php/s/WVbVyIwkAHUDaYs/download"
download_nextcloud "$DIR/poldeepner/model/pl.deduped.maca.skipgram.300.mc10.bin" "https://nextcloud.clarin-pl.eu/index.php/s/FQlYoGvXOXjnQZx/download"
download_nextcloud_raw "$DIR/poldeepner/model/kgr10.plain.skipgram.dim300.neg10.bin" "https://nextcloud.clarin-pl.eu/index.php/s/luubhnS0AvjmtQc/download?path=%2F&files=kgr10.plain.skipgram.dim300.neg10.bin"

#
# Unpack pre-trained models
#
function unpack_model_7z()
{
  MODEL=$1
  OUTPUT=`dirname $MODEL`
  if [ ! -d $MODEL ] && [ ! -f $MODEL ] ; then
    7z x "$MODEL.7z" -o$OUTPUT
    echo "- `basename $MODEL` unpacked"
  else
    echo "- `basename $MODEL` found"
  fi
}

unpack_model_7z "$DIR/poldeepner/data/nkjp-nested-simplified-v2.iob"
unpack_model_7z "$DIR/poldeepner/data/POLEVAL-NER_GOLD.json"
unpack_model_7z "$DIR/poldeepner/data/poleval2018ner-data"