import catalyst
from catalyst import dl,utils,metrics
from catalyst.contrib.losses import DiceLoss, IoULoss
import torch
from torch import nn, optim
import mlflow
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from dataset_utils import UECDataset
from augmentation_utils import get_training_augmentation,get_validation_augmentation,get_preprocessing
from model_utils import SM_preprocessing_function,SM_pytorch_model
from helper_utils import load_yaml
from pathlib import Path
from datetime import datetime

# https://github.com/catalyst-team/catalyst/blob/4e8e77f3223725355e5147af0005378e1269b115/examples/notebooks/customization_tutorial.ipynb

# SEED -----------------------------------------------------------------------------------

SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)
print(f"\ntorch: {torch.__version__},\ncatalyst: {catalyst.__version__}")

# -- Load Dataset path details -----------------------------------------------------------

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

# -----------------------------------------------------------------------------------------

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

# Model -------------------------------------------------

ACTIVATION = 'sigmoid' 
#'sigmoid' #sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
# 'softmax2d' = nn.Softmax(dim=1) # softmax in depth direction

model = SM_pytorch_model(ENCODER,
                         ENCODER_WEIGHTS,
                         CLASSES,
                         ACTIVATION)

preprocessing_fn = SM_preprocessing_function(ENCODER,
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

# Dataloader Class --------------------------------------

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)#, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)#, num_workers=4)

# Catalyst Utils ----------------------------------------

# --- Dataloader ---------

loaders = {'train':train_loader, 
           'valid':valid_loader}

# --- Criterion -----------

criterion = {"dice_loss": DiceLoss(),
             "iou_loss": IoULoss()}

# --- Optimiser -----------

optimizer_name = 'Adam'
optimizer_lr = 0.0001

if optimizer_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)

# --- Logs Hparams -------

hparams.update({'criterion':list(criterion.keys())})

hparams.update({'optimizer_name':optimizer_name,
                'optimizer_lr':optimizer_lr})

# ---- Scheduler ---------

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2])

# --- Runner -------------

#https://github.com/catalyst-team/catalyst/blob/4ca33a015998ab5602887e9d827c5e0bfca8bc64/docs/faq/multi_components.rst
#https://github.com/catalyst-team/catalyst/blob/37e7809919fb3882978409a865087cba56839fe0/tests/pipelines/test_multihead_classification.py

class CustomRunnerML2(dl.Runner):
    def handle_batch(self, batch):
        x, y = batch
        y_hat = self.model(x)
        self.batch = {"image": x,
                    "mask": y,
                    "logits": y_hat}

runner = CustomRunnerML2()

root_logdir = Path("logs")/datetime.now().strftime("%d-%m-%Y_%I-%M-%S")

runner.train(model=model,
             criterion=criterion,
             optimizer=optimizer,
             scheduler=scheduler,
             hparams=hparams,
             loaders=loaders,
             num_epochs=2,
             verbose=True,
             
             loggers={"console": dl.ConsoleLogger(log_hparams=False), 
                      "tb": dl.TensorboardLogger(logdir=root_logdir/'tb'),
                      "csv": dl.CSVLogger(logdir=root_logdir/'csv_logs'),
                      "mlflow": dl.MLflowLogger(experiment=MLFLOW_EXP,
                                                run=MLFLOW_RUN,
                                                log_batch_metrics=False,
                                                log_epoch_metrics=True)},
             
             callbacks=[dl.CriterionCallback(metric_key="dice_loss", 
                                             input_key="logits",
                                             target_key="mask",
                                             criterion_key='dice_loss'),
                        
                        dl.CriterionCallback(metric_key="iou_loss",
                                             input_key="logits", 
                                             target_key="mask",
                                             criterion_key='iou_loss'),
                        
                        dl.MetricAggregationCallback(metric_key="total_loss", 
                                                     metrics=["dice_loss", "iou_loss"], 
                                                     mode="mean"),
                        dl.SchedulerCallback(),
                        dl.BackwardCallback(metric_key="total_loss"), # can be useful if you want to use gradient clipping
                        dl.OptimizerCallback(metric_key="total_loss"),# If anything has to do with optimise
                        dl.CheckpointCallback(logdir=root_logdir/"checkpoints",
                                              loader_key="valid", 
                                              metric_key="total_loss", 
                                              minimize=True)])
