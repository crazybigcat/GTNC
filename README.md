pip install -r requirements.txt # or pip install numpy and pip install scipy


The dataset of MNIST and fashion-MNIST should be put in ./dataset/mnist/ and ./dataset/fashion or you can change para['path_dataset'] in Parameters.py. For example the file train-images-idx3-ubyte.gz downloaded from http://yann.lecun.com/exdb/mnist/ should locate on ./dataset/mnist/train-images-idx3-ubyte.gz. The fashion-MNIST can be downloaded from https://github.com/zalandoresearch/fashion-mnist.


After the dataset is well placed. You can run start_train_gtnc.py in python3 and the parameters can be changed in Parameters.py.
