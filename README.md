# PiA-LaO
Pixel-sapce Attack and Latent-sapce Optimize

## 1. Download dataset
Download ImageNet-compatible dataset from the internet and place it in the `/dataset` folder.

## 2. Train classifier
Generate watermarked images for the images in the dataset using a watermarking method. Run the `train_classifier/train.py` file to train a ResNet50 classifier to distinguish between original images and watermarked images. Please customize the weight save path.

## 3. Attack
Run the `method/attack_ImageNet.py` file to generate attacked images.
