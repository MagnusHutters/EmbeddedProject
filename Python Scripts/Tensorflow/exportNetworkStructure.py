import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Load your model
model = tf.keras.models.load_model('object_classification_model_no_bias.h5')

# Generate the plot
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

# This will save the model architecture diagram as 'model_structure.png'
