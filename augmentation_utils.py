import albumentations as albu

def get_training_augmentation(im_h,
                              im_w):
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

def get_validation_augmentation(im_h,
                                im_w):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.PadIfNeeded(min_height=im_h,
                                       min_width=im_w, 
                                       always_apply=True, 
                                       border_mode=0),
                      albu.Resize(height=im_h,width=im_w,
                                  always_apply=True)] #should be divisible by 32
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
