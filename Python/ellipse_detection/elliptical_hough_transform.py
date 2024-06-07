######################################################################
# Ellipse detection
# =================
#
# In this second example, the aim is to detect the edge of a coffee cup.
# Basically, this is a projection of a circle, i.e. an ellipse. The problem
# to solve is much more difficult because five parameters have to be
# determined, instead of three for circles.
#
# Algorithm overview
# -------------------
#
# The algorithm takes two different points belonging to the ellipse. It
# assumes that it is the main axis. A loop on all the other points determines
# how much an ellipse passes to them. A good match corresponds to high
# accumulator values.
#
# A full description of the algorithm can be found in reference [1]_.
#
# References
# ----------
# .. [1] Xie, Yonghong, and Qiang Ji. "A new efficient
#        ellipse detection method." Pattern Recognition, 2002. Proceedings.
#        16th International Conference on. Vol. 2. IEEE, 2002

import os
import argparse
import matplotlib.pyplot as plt

from skimage import io
from skimage import color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.transform import rescale, resize

def main():
    parser = argparse.ArgumentParser(description='Ellipse Hough detection.')
    parser.add_argument("-i", "--input", help='Input image')
    args = parser.parse_args()

    input_image = args.input
    print(f"input_image: {input_image}")

    # Load picture, convert to grayscale and detect edges
    filename = input_image
    image_rgb_original = io.imread(filename)
    image_rgb = resize(image_rgb_original, (image_rgb_original.shape[0] // 2, image_rgb_original.shape[1] // 2), anti_aliasing=True)
    image_gray = color.rgb2gray(image_rgb)
    edges = canny(image_gray, sigma=2.0,
                low_threshold=0.55, high_threshold=0.8)

    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                        min_size=100, max_size=400)
    print('result:', len(result))
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                    sharex=True, sharey=True)

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()

if __name__ == '__main__':
    main()
