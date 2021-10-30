# benchmark_TensorFlow_macOS
A set of Python codes and data to benchmark TensorFlow for macOS on a training task of a large CNN model for image segmentation.  
*I made this set for benchmarking TensorFlow on GPU of M1 SoC in macOS Monterey.*  

Installation Instructions of TensorFlow for GPU training in macOS Monterey:  
[Getting Started with tensorflow-metal PluggableDevice (Apple Developer)](https://developer.apple.com/metal/tensorflow-plugin/)  

Related article of mine in Medium.com:  
[Training Speed of TensorFlow in macOS Monterey](https://towardsdatascience.com/training-speed-of-tensorflow-in-macos-monterey-3b8020569be1)  


## Description
- Train.py : Main code.
- data : contains 512x512 greyscale images for training (0001.png to 0500.png) and validation (0501.png to 0600.png). Images are simple triangles just for benchmark.
- neural_networks : contains an original TF-Keras model. The model has an U-net-like structure with the total parameters of 20-Mega (20,886,706).
- utils : contains helper codes for Train.py - CSV loader, Keras callbacks, and loss and metrics.
- run_BENCHMARK_5times_5min_interval.sh / .bat : Automation scripts to perform the training task five times with five minutes interval.
- package_list : contains package lists of miniconda (miniforge) in which TensorFlow runs on GPU.

<img width="600" alt="Screenshot" src="https://user-images.githubusercontent.com/52600509/139534799-57a193c4-3114-435e-a816-056a9aedebf2.png">

## Result
I performed this benchmark code on my MacBook Air M1 and a WIndows workstation. The red circle in the above figure represents comutation time required for training data in an epoch. The yellow circle represents time required for validation.  
> Please check my Medium article for more details. [Training Speed of TensorFlow in macOS Monterey](https://towardsdatascience.com/training-speed-of-tensorflow-in-macos-monterey-3b8020569be1)  


<img width="640" alt="Table2+" src="https://user-images.githubusercontent.com/52600509/139528179-a0600ac0-044c-471a-bd3c-1acedd9dc77f.png">

