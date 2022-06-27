import numpy as np
from PIL import Image
from skimage.transform import resize
import os
import logging
import keyboard
import time
from keyboard_bindings import keyboard_numbers_binding as keyboard_binding
from tkinter import Tk, Label, Button
from tkinter import filedialog
import copy


def initialize_model_logger():
    """"
    The method below sets and initiates the Depth Model logger.
    The model logger tasks are:
    1. To print on the screen different message types (DEBUG, INFO, ERROR, CRITICAL).
    2. Write the messages in the depth_model.log which can be found in logger folder.
    (Created Function)
    """

    # Logging to specific logger which can be configure separately for each script
    logger_model = logging.getLogger(__name__)

    # Set the lowest level of "errors" that you want to see in log.
    logger_model.setLevel(logging.DEBUG)

    # Set the file where the logger messages should be written(relative path).
    dir = os.path.dirname(__file__)
    logger_path = os.path.join(dir, 'logger','depth_logger.log')

    # Set the file where the file_handler should write the messages. The "w" means that it deletes everything that
    # was written before (such that the file doesn't get too big)
    file_handler = logging.FileHandler(logger_path, "w")

    # Setting up the message format of the logger as follows:
    # 1. "%(asctime)s:" - Time and date when the message was written/printed.
    # 2. "%(levelname)s" - Text logging level for the message.
    # 3. "%(funcName)s" - Name of the function containing the logging call.
    # 4. "%(message)s" - The logged message.
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')

    # Sets the format for the file_handler
    file_handler.setFormatter(formatter)
    logger_model.addHandler(file_handler)

    # Sets the stream_handler and the format for the stream_handler (to print in the console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger_model.addHandler(stream_handler)

    return logger_model


def clear_keyboard_inputs():
    """"
    Clear the last user input (after processing it)
    (Created Function)
    """
    keyboard.unhook_all()
    keyboard.clear_all_hotkeys()
    time.sleep(0.3)


def select_pretrained_model_for_inference(logger_model):
    """"
    Interface that allows the user to select the pre-trained model on which to run inference.
    (Created Function)
    """
    print("Select the pre-trained model for inference by pressing the keyboard corresponding number (e.g press 1 for NYU):")
    print("1.NYU Depth v2 (indoor scenes)")
    print("2.KITTI (outdoor scenes)")
    while True:

        if keyboard.is_pressed(keyboard_binding[1]):
            logger_model.info("1 was pressed. Loading the NYU model...")
            model_selected_name = 'nyu.h5'
            clear_keyboard_inputs()
            break

        elif keyboard.is_pressed(keyboard_binding[2]):
            logger_model.info("2 was pressed. Loading the KITTI model...")
            model_selected_name = 'kitti.h5'
            clear_keyboard_inputs()
            break

    return model_selected_name


def load_images_using_tkinter(logger_model):
    """
    Import an image or a batch of images (.png only) using tkinter from a user selected file.
    (Created Function)
    """

    # Create the tkinter window
    window = Tk()

    # Add a Label widget
    label = Label(window, text="Press the button below to open a file from which to load the input Image(s)",font=('Aerial 11'))
    label.pack(pady=30)
    # Add the Button Widget
    Button(window, text="Load Image(s) from folder").pack()
    dir = os.path.dirname(__file__)
    # Initial directory from which to select the images
    initial_dir = os.path.join(dir, 'input_images')
    images = filedialog.askopenfilenames(parent=window, initialdir=initial_dir,
                                        title="Load input image(s) from a specific folder",
                                        filetypes=[("png files", "*.png")])

    images = list(images)
    window.destroy()
    logger_model.info("Loaded Images: " + str(images)[:])
    return images

def load_demo_images(logger_model, model_name):
    """
    (Created Function)
    Load the demo images.
    """

    images = []
    dir = os.path.dirname(__file__)
    # Output directory for images
    input_dir = os.path.join(dir, 'input_images', 'demo_images')
    if model_name == "nyu.h5":
        image_path1 = input_dir + "//people.png"
        image_path2 = input_dir + "//1_image.png"

    elif model_name == "kitti.h5":
        image_path1 = input_dir +"//000296.png"
        image_path2 = input_dir + "//000290.png"

    images = [image_path1, image_path2]

    logger_model.info("Loade Image: " + str(images)[:])

    return images

def load_images(image_files, logger_model, model_name):
    """"
    (Created function)
    -> Added the option to load images of different sizes (before all input images had to be the same size).
    -> Makes sure to convert non-RGB format images to the RGB standard format (3 channels) - Requirement for the model.
      (Palettised colored images can also be used)
    -> Resize the input image(s) to the image sizes used during training/testing (nyu - (640, 480), kitti - (1280, 384)).
       The resulting image(s) will be used as input for depth inference.
    -> Also returns the input image with original size after converting to RGB format in order to be used later as input
     for YOLO.
    """
    loaded_images = []
    loaded_images_name = []
    org_images = []
    inputs_shape = []
    for file in image_files:
        x = np.array(Image.open(file))
        parsed_file_name = file.split("/")
        input_image_name = parsed_file_name[-1]
        loaded_images_name.append(input_image_name)
        x = convert_to_rgb_format(x,file, logger_model, input_image_name)
        input_image = copy.deepcopy(x)
        org_images.append(np.stack(input_image))
        x, input_shape = resize_input_image(x, logger_model, input_image_name, model_name)
        loaded_images.append(np.stack(x))
        inputs_shape.append(input_shape)

    return loaded_images, loaded_images_name, inputs_shape, org_images


def convert_to_rgb_format(image,image_path, logger_model, input_image_name):
    """
    Makes sure to convert non-RGB format images to the RGB standard format (3 channels) - Requirement for the model.
    (Palettised colored images can also be used)
    (Created function)
    """

    # Had a case where a coloured image was "palettised" (2 channels only) - palettised.png in the input_images.
    # Hence in order to solve this issue, we first convert the image to "RGB".
    if image.ndim != 3 or image.shape[-1] != 3:
        image = np.clip(np.asarray(Image.open(image_path).convert('RGB')) / 255, 0, 1)
        logger_model.info(input_image_name + " is not in RGB format. Converting to RGB format...")
    else:
        image = np.clip(np.asarray(Image.open(image_path), dtype=float) / 255, 0, 1)
    return image

def resize_input_image(image, logger_model, input_image_name, pretrained_model):
    """"
    Resize the input image(s) to the image sizes used during training/testing (nyu - 640, 480, kitti - 1280, 384).
    The resulting image(s) will be used as input for depth inference.
    The encoder architecture also expects the image dimensions to be divisible by 32.
    Resizing method used: bicubic (to retain as many details as possible)
    (Created Function)
    """
    height , width, channels = image.shape
    input_shape = (height, width)
    is_input_image_upscaled = 0

    if pretrained_model == "nyu.h5":
        output_shape = (480, 640)
    elif pretrained_model == 'kitti.h5':
        output_shape = (384, 1280)

    if width % output_shape[1] != 0:
        image = resize(image, output_shape, order=3, preserve_range=True, mode='reflect', anti_aliasing=True)
        is_input_image_upscaled = 1

    if height % output_shape[0] != 0:
        image = resize(image, output_shape, order=3, preserve_range=True, mode='reflect', anti_aliasing=True)
        is_input_image_upscaled = 1

    if is_input_image_upscaled == 1:
        logger_model.info(
            input_image_name + " has been resized from " + str(input_shape) + " to " + str(output_shape) +
            " in order to be used as input for the depth inference.")

    return image,input_shape


def predict(model, images, logger_model, loaded_input_images_name, minDepth=10, maxDepth=1000, batch_size=1):
    """
    (Modified function)
    -> Now allows the user to load images of different sizes.
    """
    # Support multiple RGBs, one RGB image, even grayscale
    output_images = []
    i = 0
    for image in images:
        logger_model.info("Currently predicting " + loaded_input_images_name[i] + " ...")
        if len(image.shape) < 3: image = np.stack((image,image,image), axis=2)
        if len(image.shape) < 4: image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Compute predictions
        predictions = model.predict(image, batch_size=batch_size)
        # Put in expected range
        output_images.append(np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth)
        i += 1
    return output_images


def output_depth_images(outputs, logger_model, inputs_shape, inputs=None, loaded_input_images_name = None):
    """
    (Created function)
    Output the depth images along with the input images to the output_images folder.
    """
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize
    for index in range(len(outputs)):
        plasma = plt.get_cmap('plasma')
        shape = (outputs[index][0].shape[0], outputs[index][0].shape[1], 3)

        dir = os.path.dirname(__file__)
        # Output directory for images
        output_dir = os.path.join(dir, 'output_images')

        name_pic = loaded_input_images_name[index].split(".")
        name_pic = name_pic[0]

        # Output the input image
        im1 = Image.fromarray(np.uint8(inputs[index] * 255))
        im1.save(output_dir + "\\" + loaded_input_images_name[index])
        logger_model.info(loaded_input_images_name[index] + " finished processing! Image can be found in the output_images folder.")

        rescaled = outputs[index][0][:, :, 0]
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled / np.max(rescaled)

        # Output the Colored Depth Image with Original Output Size of the algorithm (w/2, h/2)
        colored_depth_image = np.uint8(plasma(rescaled)[:, :, :3] * 255)
        im2 = Image.fromarray(colored_depth_image)
        colored_depth_image_name = name_pic + "_depth_colored.png"
        im2.save(output_dir + "\\" + colored_depth_image_name)
        logger_model.info(colored_depth_image_name + " finished processing! Image can be found in the output_images folder.")

        # Output the "black/white" Depth Image with Original Output Size of the algorithm (w/2, h/2)
        depth_image = np.uint8(to_multichannel(outputs[index][0] * 255))
        im3 = Image.fromarray(depth_image)
        depth_image_name = name_pic + "_depth.png"
        im3.save(output_dir + "\\" + depth_image_name)
        logger_model.info(depth_image_name + " finished processing! Image can be found in the output_images folder.")

        # Output the Colored Depth Image(width, height are the same as the input image)
        upscaled_colored_depth_image_shape = inputs_shape[index]
        upscaled_colored_depth_image = np.uint8((resize(colored_depth_image, upscaled_colored_depth_image_shape, order=3, preserve_range=True, mode='reflect', anti_aliasing=True)))
        im4 = Image.fromarray(upscaled_colored_depth_image)
        upscaled_colored_depth_image_name = name_pic + "_upscaled_depth_colored.png"
        im4.save(output_dir + "\\" + upscaled_colored_depth_image_name)
        logger_model.info(upscaled_colored_depth_image_name + " finished processing! Image can be found in the output_images folder.")

        # Output the "black/white" Depth Image(width, height are the same as the input image)
        upscaled_depth_image_shape = inputs_shape[index]
        upscaled_depth_image = np.uint8(to_multichannel(resize(outputs[index][0], upscaled_depth_image_shape, order=3, mode='reflect', anti_aliasing=True)) * 255)
        upscaled_depth_image = upscaled_depth_image[:,:,0]
        im5 = Image.fromarray(upscaled_depth_image)
        upscaled_depth_image_name = name_pic + "_upscaled_depth.png"
        im5.save(output_dir + "\\" + upscaled_depth_image_name)
        logger_model.info(upscaled_depth_image_name + " finished processing! Image can be found in the output_images folder.")


def DepthNorm(x, maxDepth):
    return maxDepth / x


def scale_up(scale, images):
    scaled = []
    
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)


def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)
        
def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
    
    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []
        
        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:,:,0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:,:,:3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)
    
    return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))


def save_images(filename, outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    montage = display_images(outputs, inputs, is_colormap, is_rescale)
    im = Image.fromarray(np.uint8(montage*255))
    im.save(filename)


def load_test_data(test_data_zip_file='nyu_test.zip'):
    print('Loading test data...', end='')
    import numpy as np
    from data import extract_zip
    data = extract_zip(test_data_zip_file)
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print('Test data loaded.\n')
    return {'rgb':rgb, 'depth':depth, 'crop':crop}

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10

def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []
    
    for i in range(N//bs):    
        x = rgb[(i)*bs:(i+1)*bs,:,:,:]
        
        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]
        pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
        
        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            predictions.append(   (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))   )
            testSetDepths.append(   true_y[j]   )

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(predictions, testSetDepths)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    return e
