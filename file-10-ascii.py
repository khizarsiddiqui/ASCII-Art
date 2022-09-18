# first commit
# initializing the project
# In this project, you’ll use Pillow, the friendly fork of the Python Imaging Library, 
# to read in the images, access their underlying data, and create and modify them. 
# You’ll also use the numpy library to compute averages.

from multiprocessing.util import close_all_fds_except
import sys, random, argparse
import numpy as np
import math
from PIL import Image
# Defining the Grayscale Levels and Grid (step 1)
# a ramp (an increasing set of values) of ASCII characters to represent grayscale values in the range [0, 255].

cols=80
scale=0.43

# 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`". "
# 10 levels of gray
gscale2 = "@%#*+=-:. "
# open the image and convert to grayscale
image = Image.open(fileName).convert("L")
# store the image dimensions
W, H = image.size[0], image.size[1]
# compute the tile width
w = W/cols
# compute the tile height based on the aspect ratio and scale of the font
h = w/scale
# compute the number of rows to use in the final grid
rows = int(H/h)

# Computing the Average Brightness (step 2)
def getAverageL(image):
# get the image as a numpy array
    im = np.array(image)
# get the dimensions
    w,h = im.shape
# get the average
    return np.average(im.reshape(w*h))

# Generating the ASCII Content from the Image (step 3)
# an ASCII image is a list of character strings
aimg = []