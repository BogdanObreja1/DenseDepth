import os
import argparse



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import initialize_model_logger, select_pretrained_model_for_inference, load_images_using_tkinter, load_images, predict,  output_depth_images
from matplotlib import pyplot as plt


# Initializing the logger
logger_model = initialize_model_logger()

# UI that allows the user to select the pre-trained model on which to run inference.
model_name = select_pretrained_model_for_inference(logger_model)

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default=model_name, type=str, help='Trained Keras model file.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

logger_model.info('Model loaded ({0}).'.format(args.model))

# Allow the user to load the input images
images = load_images_using_tkinter(logger_model)

while not images:
    logger_model.error("No images selected! Please try again.")
    images = load_images_using_tkinter(logger_model)

# Input images. The functions has been modified to load images of different sizes.
# Check load_images in utils.py
inputs, loaded_input_images_name = load_images(images, logger_model)

# Compute results
outputs = predict(model, inputs, logger_model, loaded_input_images_name)

# Output in the output_images folder the input/depth image
output_depth_images(outputs.copy(), logger_model, inputs.copy(), loaded_input_images_name)
