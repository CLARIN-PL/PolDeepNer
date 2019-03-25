#!/bin/bash

ROOT=`pwd`

docker run -it --entrypoint /bin/bash -v $ROOT/poldeepner:/poldeepner poldeepner