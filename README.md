# Fast Autoaugment
<img src="faa.png">

## Usage
### Training
#### CIFAR10
```bash
# ResNet20 (w/o FastAutoAugment)
python train.py --seed=24 --scale=3 --optimizer=sgd --fast_auto_augment=False

# ResNet20 (w/ FastAutoAugment)
python train.py --seed=24 --scale=3 --optimizer=sgd --fast_auto_augment=True

# ResNet20 (w/ FastAutoAugment, Pre-searched policy)
python train.py --seed=24 --scale=3 --optimizer=sgd --fast_auto_augment=True \
                --augment_path=runs/ResNet_Scale3_FastAutoAugment/augmentation.cp

# ResNet32 (w/o FastAutoAugment)
python train.py --seed=24 --scale=5 --optimizer=sgd --fast_auto_augment=False

# ResNet32 (w/ FastAutoAugment)
python train.py --seed=24 --scale=5 --optimizer=sgd --fast_auto_augment=True

# EfficientNet (w/ FastAutoAugment)
python train.py --seed=24 --pi=0 --optimizer=adam --fast_auto_augment=True \
                --network=efficientnet_cifar10 --activation=swish
```

#### ImageNet (You can use any backbone networks in [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html))
```bash

# BaseNet (w/o FastAutoAugment)
python train.py --seed=24 --dataset=imagenet --optimizer=adam --network=resnet50

# EfficientNet (w/ FastAutoAugment) (UnderConstruction)
python train.py --seed=24 --dataset=imagenet --pi=0 --optimizer=adam --fast_auto_augment=True \
                --network=efficientnet --activation=swish
```

### Eval
``` 
python eval.py --model_path=runs/ResNet_Scale3_Basline
```

## Experiments
### Fast AutoAugment
#### ResNet20 (CIFAR10)
<img src="res20_valid.png">

#### ResNet34 (CIFAR10)
<img src="res34_valid.png">

### Augmented images

### Searched policy
