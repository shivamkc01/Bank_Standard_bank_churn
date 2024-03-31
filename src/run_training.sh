#!/bin/bash

set -e
echo "TRAINING STARTED!"
echo "Aurthur by Shivam Chhetry on 30.03.24 @ 23.08 PM"

python train.py --fold 10 --model dt --logs DecisionTree_with_smote
echo "SUCCESSFUL DONE!"