#!/bin/bash

docker run -it --expose 127.0.0.1:500:500/tcp --env COMET_API_KEY=$COMET_API_KEY ift6758/serving:1.0.0