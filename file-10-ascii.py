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

# grayscale level values from:
# http://paulbourke.net/dataformats/asciiart/

# 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'."
# 10 levels of gray
gscale2 = "@%#*+=-:. "

# Computing the Average Brightness (step 2)
def getAverageL(image):
# get the image as a numpy array
    im = np.array(image)
# get the dimensions
    w,h = im.shape
# get the average
    return np.average(im.reshape(w*h))

def convertImageToAscii(fileName, cols, scale, moreLevels):
    # declare globals
    global gscale1, gscale2
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
    print("cols: %d, rows: %d" % (cols, rows))
    print("tile dims: %d x %d" % (w, h))
# check if image size is too small
    if cols > W or rows > H:
        print("Image too small for specified cols!")
        exit(0)
# Generating the ASCII Content from the Image (step 3)
# an ASCII image is a list of character strings
    aimg = []
# generate the list of tile dimensions
    for j in range(rows):
        y1 = int(j*h)
        y2 = int((j+1)*h)
        # correct the last tile
        if j == rows-1:
            y2 = H
# append an empty string
        aimg.append("")
        for i in range(cols):
# crop the image to fit the tile
            x1 = int(i*w)
            x2 = int((i+1)*w)
# correct the last tile
            if i == cols-1:
                x2 = W
# crop the image to extract the tile into another Image object
            img = image.crop((x1, y1, x2, y2))
# get the average luminance
            avg = int(getAverageL(img))
# look up the ASCII character for grayscale value (avg)
            if moreLevels:
                gsval = gscale1[int((avg*69)/255)]
            else:
                gsval = gscale2[int((avg*9)/255)]
# append the ASCII character to the string
            aimg[j] += gsval

# return text image
    return aimg

# main function
def main():
# Command line operators (step 4)
    descStr = "This program converts an image into ASCII Art."
    parser = argparse.ArgumentParser(description="descStr")
# add expected arguments
    parser.add_argument('--file', dest='imgFile', required=True)
    parser.add_argument('--scale', dest='scale', required=False)
    parser.add_argument('--out', dest='outFile', required=False)
    parser.add_argument('--cols', dest='cols', required=False)
    parser.add_argument('--morelevels', dest='moreLevels', action='store_true')

# parse arguments
    args = parser.parse_args()

    imgFile = args.imgFile
# set output file
    outFile = "out.txt"
    if args.outFile:
        outFile = args.outFile
# set scale default as 0.43, which suits a Courier font
    scale = 0.43
    if args.scale:
        scale = float(args.scale)
# set cols
    cols = 80
    if args.cols:
        cols = int(args.cols)
        
    print('generating ASCII art ...')
# convert image to ASCII text
    aimg = convertImageToAscii(imgFile, cols, scale, args.moreLevels)
# Writing the ASCII Art Strings to a Text File (step 5)
# open a new text file
    f = open(outFile, 'w')
# write each string in the list to the new file
    for row in aimg:
        f.write(row + '\n')
# clean up
    f.close()
    print("ASCII art written to %s" % outFile)

# call main
if __name__ == '__main__':
    main() 