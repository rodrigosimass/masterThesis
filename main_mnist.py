from util.mnist.tools import *
from util.mnist.plot import *

image_size = [28, 28]
data, labels, _, _ = read_mnist(dim=image_size, n_train=60000)

viz_activity(data, labels)
viz_class_activity_2D(data, labels)
viz_class_activity_1D(data, labels)
viz_class_activity_1D(data, labels, sorted=True)
