# Image Classification and Generation

These are my weekly tasks from the Deep Learning for Cognitive Computing for Developers course (TIES4911) at the University of Jyväskylä, from January to April 2026. Most tasks are Jupyter notebooks, and some are Python scripts and small Flask or Gradio apps. I showed every task in a short live demo in class.

## What is inside

| Folder | What it does |
|---|---|
| `Task 1` | A TensorFlow computation graph in TensorBoard. Linear regression on a small student scores dataset with scikit-learn, low-level TensorFlow and Keras, compared together. Demo 1: car price prediction with a small neural network |
| `Task 2` | Fashion-MNIST classifiers compared. A classifier for the scikit-learn Wine dataset. Predictions and TensorBoard curves. A small network that learns XOR. Demo 2: the car price model in TensorFlow |
| `Task 3` | Autoencoders on Fashion-MNIST (vanilla, deep, sparse, denoising, convolutional and dense). CNNs on Fashion-MNIST with dropout, batch normalization and augmentation. A CNN on CIFAR-10. Demo 3: image classification on a Kaggle dataset |
| `Task 4` | Pretrained Keras models compared on sample images. Transfer learning on a flower dataset (frozen and fine-tuned). A MobileNetV2 vehicle classifier (bike, bus and car) with a Flask REST API. A multi-label vehicle model with its own service. Video classification on a UCF101 subset (frames, CNN and temporal pooling) |
| `Task 5` | Object detection and segmentation (YOLOv8, Faster R-CNN, DeepLabV3 and U-Net) and neural style transfer |
| `Task 8` | Generative models on CIFAR-10: a conditional VAE, a DCGAN and an MLP GAN. Stable Diffusion with KerasCV, and pix2pix on edges2shoes. Sample outputs are in every `results` folder |
| `Task 9` | A Gradio app to test prompt techniques (original, zero-shot, one-shot and few-shot) that saves the results to CSV |

Tasks 6, 7 and 10 are not here, as I only kept them as submitted zip files or instructions.

## Not added

- Datasets, downloaded images, trained models (`.h5`, `.pt`) and virtual environments. They are too big, and most of the images came from the web.
- The image outputs in the 2 Task 5 notebooks and the Demo 3 notebook, for the same reason. The text outputs are still there.
- Course instructions and lecture slides.

Some notebooks and services use full paths from my laptop (`E:\JYU\...`). Change them before running.
