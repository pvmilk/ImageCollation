#!/bin/bash - 
#===============================================================================
#
#          FILE: batch-ft.sh
# 
#         USAGE: ./batch-ft.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 03/02/2022 00:12
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

DATA0='P1'
DATA1='P3'

WEIGHTS=('resnet18' 'resnet50' 'resnet18ft0' 'resnet18ft1' 'resnet50ft1')

for weight in "${WEIGHTS[@]}"; do
    echo '### running on weight : '${weight}
    mkdir -p ${DATA0}-${DATA1}/${weight}

    python3 main.py \
        --manuscript1 ~/data/img-collation/${DATA0}/illustration \
        --manuscript2 ~/data/img-collation/${DATA1}/illustration \
        --results_dir ${DATA0}-${DATA1}/${weight} \
        --weight ${weight} \
        --ground_truth ground_truth/${DATA0}-${DATA1}.json > ${DATA0}-${DATA1}/${weight}/results.log
done
