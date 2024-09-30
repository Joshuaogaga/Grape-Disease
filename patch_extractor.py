# patch_extractor.py

import tensorflow as tf

# Define the custom PatchExtractor layer
@tf.keras.utils.register_keras_serializable()
class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):  # Accept **kwargs to handle unexpected arguments
        super(PatchExtractor, self).__init__(**kwargs)  # Pass **kwargs to the parent class
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches