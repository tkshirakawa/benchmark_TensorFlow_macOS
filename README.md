# benchmark_TensorFlow_macOS
A set of Python codes and data to benchmark TensorFlow for macOS on a training task of a large CNN model for image segmentation.  
*I made this set for benchmarking TensorFlow on GPU of M1 SoC in macOS Monterey.*  

Installation Instructions of TensorFlow for GPU training in macOS Monterey:  
[Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/) (Apple Developer)  


## Description
- Train.py : Main code.
- data : contains 512x512 greyscale images for training (0001.png to 0500.png) and validation (0501.png to 0600.png). *Images are simple triangles just for benchmark.*
- neural_networks : contains a TF-Keras model. 
- utils : contains helper codes for Train.py - CSV loader, Keras callbacks, and loss and metrics. 
- run_BENCHMARK_5times_5min_interval.sh / .bat : Automation scripts to perform the training task five times with five minutes interval.


## Result



<img width="750" alt="Screen Shot 2021-10-30 at 17 19 49" src="https://user-images.githubusercontent.com/52600509/139525861-4cbb1c9e-9f5f-4b98-ac6c-74da0689813f.png">
