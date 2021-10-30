# benchmark_TensorFlow_macOS
Python code to benchmark TensorFlow for macOS


## Description
- Train.py : Main code.
- data : contains 512x512 greyscale images for training (0001.png to 0500.png) and validation (0501.png to 0600.png). Images are simple triangles just for benchmark.
- neural_networks : contains a TF-Keras model. 
- utils : contains helper codes for Train.py - CSV loader, Keras callbacks, and loss and metrics. 
- run_BENCHMARK_5times_5min_interval.sh / .bat : Automation scripts to perform the training task five times with five minutes interval.


