import catalyst
from catalyst import utils
from catalyst import metrics
from catalyst.contrib.losses import DiceLoss, IoULoss
import torch
from torch import nn
import numpy as np
import segmentation_models_pytorch as smp
import yaml
from yaml.loader import SafeLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import os,glob,cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from catalyst import dl
from datetime import datetime
from pathlib import Path

SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

print(f"\ntorch: {torch.__version__},\ncatalyst: {catalyst.__version__}")

def load_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.load(f,Loader=SafeLoader)
    return data

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

im_w = 320
im_h = 320

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, 
                              rotate_limit=0.4, 
                              shift_limit=0.1, 
                              p=0.4, 
                              border_mode=0),
        
        albu.PadIfNeeded(min_height=im_h,
                         min_width=im_w, 
                         always_apply=True, 
                         border_mode=0),
        
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.3),
        
        albu.OneOf([albu.CLAHE(p=1),
                    albu.RandomBrightnessContrast(p=1),
                    albu.RandomGamma(p=1)],
                   p=0.6),
        
        albu.OneOf([albu.Sharpen(p=1),
                    albu.Blur(blur_limit=3, p=1),
                    albu.MotionBlur(blur_limit=3, p=1)],
                   p=0.6),
        
        albu.OneOf([albu.RandomBrightnessContrast(p=1),
                    albu.HueSaturationValue(p=1)],p=0.6),
        
        albu.Resize(height=im_h,width=im_w,always_apply=True)]
    
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.PadIfNeeded(min_height=im_h,
                                       min_width=im_w, 
                                       always_apply=True, 
                                       border_mode=0),
                      albu.Resize(height=im_h,width=im_w,always_apply=True)] #should be divisible by 32
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class Dataset(BaseDataset):
    """UEC Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    CLASSES = CLASSES
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,):
        
        self.ids = os.listdir(images_dir)[:60]
        self.ids = [os.path.splitext(i)[0] for i in self.ids]
        self.images_fps = [os.path.join(images_dir, image_id + '.jpg') for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id + '.png') for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls) + 1 for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # Read Data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])[:,:,-1]
        
        # extract certain classes from mask 
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # apply augmentations
        
        if type(self.augmentation) != type(None):
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 
                                                     ENCODER_WEIGHTS)

# Dataset Class -----------------------------------------
train_dataset = Dataset(train_image_folder, 
                        train_mask_folder,
                        augmentation=get_training_augmentation(), 
                        preprocessing=get_preprocessing(preprocessing_fn),
                        classes=CLASSES)

valid_dataset = Dataset(test_image_folder, 
                        test_mask_folder, 
                        augmentation=get_validation_augmentation(), 
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
DEVICE = 'cuda'

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
    logdir=Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S"),
    num_epochs=2,
    verbose=True
)

