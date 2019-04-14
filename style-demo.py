from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from scipy.optimize import fmin_l_bfgs_b
from imageio import imwrite
from tqdm import tqdm
import styledemo_utils as sdu
import sys


'''
Set Variables.
'''

#Get user input.
target_file = sys.argv[1]           #Path to target image
style_file = sys.argv[2]            #Path to style image

#Define weights for loss components.
style_weight = 1                    #Weight of image style
content_weight = 0.025              #Weight of image content
total_variation_weight = 10 ** -4   #Weight of image variation
out_h = 400                         #Output image height in pixels
iterations = 20                     #Number of transfer interations


'''
Prepare images, model, and loss function.
'''

#Calculate width to scale output image to a height of out_h.
w, h = load_img(target_file).size
out_w = int(w * out_h / h)

#Load pre-processed images and prepare placeholder.
target = K.constant(sdu.preprocess_image(target_file, out_h, out_w))
style = K.constant(sdu.preprocess_image(style_file, out_h, out_w))
combined = K.placeholder((1, out_h, out_w, 3))

#Combine images into a single tensor.
input_tensor = K.concatenate([target, style, combined], axis = 0)

#Load VGG19 network with ImageNet weights.
model = vgg19.VGG19(input_tensor = input_tensor, weights = "imagenet",
                    include_top = False)

#Define general loss function.
weights = [content_weight, style_weight, total_variation_weight]
loss = sdu.general_loss(model, weights, combined, out_h, out_w)

#Get gradients of the generated image.
grads = K.gradients(loss, combined)[0]

#Function to get the values of the current loss and the current gradients.
get_loss_and_grads = K.function([combined], [loss, grads])

#Prepare image file for processing.
x = sdu.preprocess_image(target_file, out_h, out_w)
x = x.flatten()

#Initiate evaluator.
evaluator = sdu.evaluator()


'''
Processing loop
'''

print("Processing image...\n\n")
for i in tqdm(range(iterations)):

    #Perform style transfer.
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     args = (get_loss_and_grads, out_h, out_w),
                                     fprime = evaluator.grads, maxfun = 20)
    print("\n\nCurrent loss value:", min_val)
    
    #Post-process image.
    img = x.copy().reshape((out_h, out_w, 3))
    img = sdu.postprocess_image(img)

    #Save generated image.
    outfile = "./output/transfer_" + str(i + 1) + ".png"
    imwrite(outfile, img)
    print("Image file saved!\n")
