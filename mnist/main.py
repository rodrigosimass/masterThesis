from util import read_mnist
from viz import viz_class_activity_2D, viz_class_activity_1D, viz_activity


image_size = [28, 28]
data, labels, _, _ = read_mnist(dim=image_size, n_train=60000)

print(data.shape)
print(labels.shape)

viz_activity(data, labels)
viz_class_activity_2D(data, labels)
viz_class_activity_1D(data, labels)
