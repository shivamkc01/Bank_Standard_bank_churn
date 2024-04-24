#!/bin/bash
set -e
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}
log "TRAINING STARTED!"
log "Author: Shivam Chhetry"

python train.py --fold 10 --model lr --auc_plot False --logs lr_with_quantile --metric roc_auc || { log "Error: Training script failed"; exit 1; }
# log "#############################################################################"
# python train.py --fold 10 --model dt --logs rf_with_created_feature --auc_plot False --metric f1_score || { log "Error: Training script failed"; exit 1; }
# log "#############################################################################"
# python train.py --fold 10 --model dt --logs rf_with_created_feature --auc_plot False --metric accuracy || { log "Error: Training script failed"; exit 1; }
# log "#############################################################################"
echo "SUCCESSFUL DONE!"