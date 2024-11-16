## Code Description

This project implements a deep learning pipeline for hand gesture recognition, utilizing **TensorFlow** and **Keras**. Below is a comprehensive overview of the code:

---

### **1. Data Preprocessing**
The code prepares the image dataset for training and validation. The main preprocessing steps include:
- **Resizing Images**: All images are resized to **128x128** for uniform input size.
- **Normalization**: Pixel values are scaled to a range of 0-1 for better convergence during training.
- **Grayscale Conversion**: Images are converted to grayscale for simplicity and computational efficiency.

Example preprocessing function:
```python
def preprocess_photo(photo_path):
    img_string = tf.io.read_file(photo_path)
    img = tf.image.decode_image(img_string, channels=1)
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    return img
