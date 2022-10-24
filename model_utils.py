import segmentation_models_pytorch as smp

def SM_preprocessing_function(encoder_name: str,
                              encoder_weights: str):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, 
                                                         encoder_weights)
    return preprocessing_fn

def SM_pytorch_model(encoder_name: str,
               encoder_weights: str,
               classes_list,
               activation):
    '''
    encoder_name : Encoder Name. (e.g : 'se_resnext50_32x4d')
    encoder_weights : Pretrained weights name. (e.g : 'imagenet'')
    classes_list : List of all the class names.
    activation : Activation of the final convolution layer.
    '''
    print(f'\nEncoder : {encoder_name}\n')

    model = smp.FPN(encoder_name=encoder_name, 
                    encoder_weights=encoder_weights, 
                    classes=len(classes_list), 
                    activation=activation)
    return model