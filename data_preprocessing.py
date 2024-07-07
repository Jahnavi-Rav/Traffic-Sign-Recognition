import deeplake
import numpy as np
from skimage.color import rgb2lab
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# Set constants
N_CLASSES = 43
RESIZED_IMAGE = (32, 32)

# Load datasets from Deeplake
ds_train = deeplake.load("hub://activeloop/gtsrb-train")
ds_test = deeplake.load("hub://activeloop/gtsrb-test")

def process_image(img, resize_to):
    img = rgb2lab(img / 255.0)[:,:,0]
    img = resize(img, resize_to, mode='reflect')
    img = img.astype(np.float32)
    img = img[:, :, np.newaxis]  # Add channel dimension
    return img

def process_label(label, n_labels):
    label = int(label)
    one_hot_label = np.zeros((n_labels,), dtype=np.float32)
    one_hot_label[label] = 1.0
    return one_hot_label

def process_deeplake_sample(sample, resize_to, n_labels):
    img = sample['images'].numpy()
    label = sample['labels'].numpy()
    img = process_image(img, resize_to)
    label = process_label(label, n_labels)
    return img, label

def deeplake_to_numpy(ds, resize_to, n_labels):
    images = []
    labels = []
    for sample in ds:
        img, label = process_deeplake_sample(sample, resize_to, n_labels)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Convert Deeplake dataset to NumPy arrays
X_train, y_train = deeplake_to_numpy(ds_train, RESIZED_IMAGE, N_CLASSES)
X_test, y_test = deeplake_to_numpy(ds_test, RESIZED_IMAGE, N_CLASSES)

# Save datasets
np.savez_compressed('data/train_data.npz', X_train=X_train, y_train=y_train)
np.savez_compressed('data/test_data.npz', X_test=X_test, y_test=y_test)
