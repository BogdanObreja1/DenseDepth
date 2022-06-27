import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import initialize_model_logger, select_pretrained_model_for_inference, load_images_using_tkinter, load_images, predict,  output_depth_images

"""
Written almost fromm scratch.
"""


# Initializing the logger
logger_model = initialize_model_logger()

# UI that allows the user to select the pre-trained model on which to run inference.
depth_model_name = select_pretrained_model_for_inference(logger_model)

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Load depth model into GPU / CPU
depth_model = load_model(depth_model_name, custom_objects=custom_objects, compile=False)

logger_model.info('Model loaded ({0}).'.format(depth_model_name))

# Allow the user to load the input images
images = load_images_using_tkinter(logger_model)

while not images:
    logger_model.error("No images selected! Please try again.")
    images = load_images_using_tkinter(logger_model)

# Input images. The functions has been modified to load images of different sizes.
model_inputs, loaded_input_images_name, inputs_shape, org_imgs = load_images(images, logger_model, depth_model_name)

# Predict the depth images.
outputs = predict(depth_model, model_inputs, logger_model, loaded_input_images_name)

# Output in the output_images folder the input/depth image
output_depth_images(outputs.copy(), logger_model, inputs_shape, org_imgs.copy(), loaded_input_images_name)
