#!/bin/bash

docker exec --user docker_geotransformer -it ${USER}_geotransformer \
    /bin/bash -c "cd /home/docker_geotransformer; echo ${USER}_geotransformer container; echo ; /bin/bash"
