# Copyright (c) Meta Platforms, Inc. and affiliates.

# link to the dataset folder, model weights and the config file.
export DETECTRON2_DATASETS="/home/shoval/Documents/Repositories/CutLER/datasets"
model_weights="http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth"
config_file="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"
num_gpus=1

echo "========== start evaluating the model on dota dataset =========="

test_dataset='dota_gsd_02_test'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} --no-segm \
  --eval-only MODEL.WEIGHTS ${model_weights}

echo "========== evaluation is completed =========="