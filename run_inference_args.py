import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import initialize_model_logger, load_demo_images, load_images_using_tkinter, load_images, predict,  output_depth_images
import argparse

"""
Written almost fromm scratch.
"""

# Argument Parser
parser = argparse.ArgumentParser(description='Image Depth Estimator')
parser.add_argument('--model', required=True, choices=['nyu.h5', 'kitti.h5'], type=str, help='Choose the trained model (nyu.h5 or kitti.h5).')
parser.add_argument('--mode', required=True, choices=['demo', 'full'], type=str, help = "Allow the user to select between the following 2 modes:"
                                                                                          " 1. demo - loads a specific image as input;"
                                                                                          " 2. full - allow the user to select images from any folder as input;")
args = parser.parse_args()

# Initializing the logger
logger_model = initialize_model_logger()

# Loads the depth model name
depth_model_name = args.model

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Load depth model into GPU / CPU
depth_model = load_model(depth_model_name, custom_objects=custom_objects, compile=False)

logger_model.info('Model loaded ({0}).'.format(depth_model_name))

if args.mode == "demo":
    # Loads the demo images.
    images = load_demo_images(logger_model, args.model)


elif args.mode == "full":
    # Allow the user to load the input images from a file
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
