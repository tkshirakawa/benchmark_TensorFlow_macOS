# benchmark_TensorFlow_macOS
A set of Python codes and data to benchmark TensorFlow for macOS on a training task of a large CNN model for image segmentation.  
*I made this set for benchmarking TensorFlow on GPU of M1 SoC in macOS Monterey.*  

Installation Instructions of TensorFlow for GPU training in macOS Monterey:  
(Apple Developer) [Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/)  

Related article in Medium.com:  
(Medium) [Training Speed of TensorFlow in macOS Monterey - GPU training in M1 SoC comparing with results in Quadro RTX6000 and estimation in M1 Max SoC -](https://towardsdatascience.com/training-speed-of-tensorflow-in-macos-monterey-3b8020569be1)  


## Description
- Train.py : Main code.
- data : contains 512x512 greyscale images for training (0001.png to 0500.png) and validation (0501.png to 0600.png). Images are simple triangles just for benchmark.
- neural_networks : contains an original TF-Keras model. The model has an U-net-like structure with the total parameters of 20-Mega (20,886,706).
- utils : contains helper codes for Train.py - CSV loader, Keras callbacks, and loss and metrics.
- run_BENCHMARK_5times_5min_interval.sh / .bat : Automation scripts to perform the training task five times with five minutes interval.
- package_list : contains package lists of miniconda (miniforge) in which TensorFlow runs on GPU.

<img width="500" alt="Screenshot" src="https://user-images.githubusercontent.com/52600509/139525861-4cbb1c9e-9f5f-4b98-ac6c-74da0689813f.png">

## Result
<img width="640" alt="Table2+" src="https://user-images.githubusercontent.com/52600509/139528179-a0600ac0-044c-471a-bd3c-1acedd9dc77f.png">

