#!/bin/bash
set -e
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}
log "TRAINING STARTED!"
log "Author: Shivam Chhetry"

python train.py --fold 10 --model lr --auc_plot True --metric roc_auc || { log "Error: Training script failed"; exit 1; }

echo "SUCCESSFUL DONE!"