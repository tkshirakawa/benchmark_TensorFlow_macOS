# benchmark_TensorFlow_macOS
A set of Python codes and data to benchmark TensorFlow with a training task of a large CNN model for image segmentation.
I made this set for benchmarking TensorFlow on GPU of M1 SoC in macOS Monterey.


Installation Instructions:
[TensorFlow for GPU training in macOS Monterey](https://developer.apple.com/metal/tensorflow-plugin/)


## Description
- Train.py : Main code.
- data : contains 512x512 greyscale images for training (0001.png to 0500.png) and validation (0501.png to 0600.png). Images are simple triangles just for benchmark.
- neural_networks : contains a TF-Keras model. 
- utils : contains helper codes for Train.py - CSV loader, Keras callbacks, and loss and metrics. 
- run_BENCHMARK_5times_5min_interval.sh / .bat : Automation scripts to perform the training task five times with five minutes interval.


