import catalyst
from catalyst import utils
from catalyst import metrics
from catalyst.contrib.losses import DiceLoss, IoULoss
import torch
from torch import nn
import numpy as np
import segmentation_models_pytorch as smp
import mlflow
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import os,glob,cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from catalyst import dl
from datetime import datetime
from pathlib import Path
from dataset_utils import UECDataset
from augmentation_utils import get_training_augmentation,get_validation_augmentation,get_preprocessing
from helper_utils import load_yaml

SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

print(f"\ntorch: {torch.__version__},\ncatalyst: {catalyst.__version__}")

# -- Load Dataset path details ----------------------------------
dataset_details = load_yaml(r'dataset_details.yaml')

data_root = dataset_details['DATA_ROOT']
category_txt = dataset_details['category_txt']
train_names_txt = dataset_details['train_names_txt']
test_names_txt = dataset_details['test_names_txt']
CLASSES = dataset_details['CLASSES']

train_image_folder = os.path.join(data_root, 'train', 'img')
train_mask_folder = os.path.join(data_root, 'train', 'mask')
test_image_folder = os.path.join(data_root, 'test', 'img')
test_mask_folder = os.path.join(data_root, 'test', 'mask')

# -------------------------------------------------------------

im_w = 320
im_h = 320

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = 'cuda'
MLFLOW_EXP = 'experiment1'
MLFLOW_RUN = 'run3'
print(f'MLFlow tracking uri : {mlflow.get_tracking_uri()}')

hparams = {'ENCODER':ENCODER,
           'ENCODER_WEIGHTS':ENCODER_WEIGHTS,
           'MLFLOW_EXP':MLFLOW_EXP,
           'MLFLOW_RUN':MLFLOW_RUN,
           'Image_Width':im_w,
           'Image_Height':im_h,
           'Train_Folder':{'ImageFolder':train_image_folder,
                           'MaskFolder':train_mask_folder},
           'Validation_Folder':{'ImageFolder':test_image_folder,
                           'MaskFolder':test_mask_folder}}

# ------------------------------------------------------------

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 
                                                     ENCODER_WEIGHTS)

# Dataset Class -----------------------------------------
train_dataset = UECDataset(train_image_folder,
                           train_mask_folder,
                           augmentation=get_training_augmentation(im_w=im_w,
                                                                  im_h=im_h),
                           preprocessing=get_preprocessing(preprocessing_fn),
                           classes=CLASSES)

valid_dataset = UECDataset(test_image_folder, 
                           test_mask_folder, 
                           augmentation=get_validation_augmentation(im_w=im_w,
                                                                    im_h=im_h), 
                           preprocessing=get_preprocessing(preprocessing_fn),
                           classes=CLASSES)

# Dataloader Class -------------------------------------

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)#, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)#, num_workers=4)

loaders = {'train':train_loader, 
           'valid':valid_loader}

# Model -------------------------------------------------
ENCODER = ENCODER 
ENCODER_WEIGHTS = ENCODER_WEIGHTS
CLASSES = CLASSES
ACTIVATION = None 
#'sigmoid' #sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
# 'softmax2d' = nn.Softmax(dim=1) # softmax in depth direction

print(f'\nEncoder : {ENCODER}\n')

model = smp.FPN(encoder_name=ENCODER, 
                encoder_weights=ENCODER_WEIGHTS, 
                classes=len(CLASSES), 
                activation=ACTIVATION)

criterion = {"dice": DiceLoss(),"iou": IoULoss()}

callbacks = [dl.BatchTransformCallback(scope="on_batch_end",
                                       transform=nn.Sigmoid(),
                                       input_key="logits",
                                       output_key="pred_mask"),
             dl.CriterionCallback("pred_mask", "mask", "loss_dice", criterion_key="dice"),
             dl.CriterionCallback("pred_mask", "mask", "loss_iou", criterion_key="iou"),
             dl.MetricAggregationCallback("loss",mode="weighted_sum",metrics={"loss_dice": 1.0, "loss_iou": 1.0})]

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

runner = dl.SupervisedRunner(input_key="image", 
                             target_key="mask", 
                             output_key="logits")

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    callbacks=callbacks,
    hparams=hparams,
    loggers={"mlflow": dl.MLflowLogger(experiment=MLFLOW_EXP,
                                       run=MLFLOW_RUN,
                                       log_batch_metrics=True,
                                       log_epoch_metrics=True)},
    logdir=Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S"),
    num_epochs=2,
    verbose=True
)

