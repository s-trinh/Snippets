#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import sklearn
from sklearn.cluster import DBSCAN
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    print(f"HDBSCAN is not available (> 1.3)")

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # needed for 3D plot
from datetime import datetime

import cv2 as cv

def subsample(img, factor):
    if factor > 1:
        return cv.resize(img, (img.shape[1]//factor, img.shape[0]//factor), interpolation=cv.INTER_AREA)
    else:
        return img

def upsample(img, factor):
    if factor > 1:
        return cv.resize(img, (img.shape[1]*factor, img.shape[0]*factor))
    else:
        return img

def computeHSV(cropped_image, mask=None, verbose=True):
    hsv = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)
    # Compute some stats (median, std) on the different HSV components
    # see doc on uchar HSV on OpenCV for the component ranges
    h_median = 2 * np.median(hsv[:,:,0])
    s_median = 100 * np.median(hsv[:,:,1]) / 255.0
    v_median = 100 * np.median(hsv[:,:,2]) / 255.0
    h_std = 2 * np.std(hsv[:,:,0])
    s_std = 100 * np.std(hsv[:,:,1]) / 255.0
    v_std = 100 * np.std(hsv[:,:,2]) / 255.0
    if verbose:
        print(f"HSV={hsv.shape} ; dtype={hsv.dtype}")
        print(f"H median={h_median} ; H std={h_std} ; S median={s_median} ; S std={s_std} ; V median={v_median} ; V std={v_std}")

    # Keep only HSV data into the mask
    hue_masked = cv.copyTo(hsv[:,:,0], mask)
    saturation_masked = cv.copyTo(hsv[:,:,1], mask)
    value_masked = cv.copyTo(hsv[:,:,2], mask)

    # Try to get the regions excluded from the mask
    inverse_mask = 255*np.ones(hue_masked.shape, dtype=np.uint8)
    if mask is not None:
        inverse_mask = inverse_mask - mask

    return hsv, hue_masked, saturation_masked, value_masked, inverse_mask

def getLinear(x, xa, xb, ya=0, yb=255):
    if xb - xa > 0:
        alpha = (yb - ya) / (xb - xa)
        beta = yb - alpha * xb
        return alpha * x + beta
    else:
        return 0

# Apply a colormap for the different labels
def visuLabels(labels_reshape, n_clusters_, colormap):
    labels_visu = np.zeros(labels_reshape.shape, dtype=np.uint8)

    if n_clusters_ > 1:
        labels_visu = np.uint8(255 / float(n_clusters_) * labels_reshape)

    labels_visu_colormap = cv.applyColorMap(labels_visu, colormap)
    return labels_visu_colormap

# https://stackoverflow.com/questions/35355930/figure-to-image-as-a-numpy-array/57988387#57988387
def imageFromPlot(fig, ax, padding=0):
    fig.tight_layout(pad=padding)
    ax.margins(0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot_bgr = cv.cvtColor(image_from_plot, cv.COLOR_RGB2BGR)

    return image_from_plot_bgr

def main():
    parser = argparse.ArgumentParser(description='DBSCAN.')
    parser.add_argument("-i", "--input", help='Input image.')
    parser.add_argument("--subsample", type=int, default=1, help='Subsample factor (e.g. 2, 4, ...).')
    parser.add_argument("--eps", type=float, default=10, help='DBSCAN: eps parameter.')
    parser.add_argument("--min-samples", type=int, default=1, help='DBSCAN: min_samples parameter.')
    parser.add_argument("-s", "--save", "--output", default="", help='Save images.')
    parser.add_argument("--no-roi", action='store_true', help='No ROI.')
    parser.add_argument("--roi", type=int, nargs='+', help='ROI.')
    parser.add_argument("--HSV-min", type=int, nargs='+', help='.')
    parser.add_argument("--HSV-max", type=int, nargs='+', help='.')
    parser.add_argument("--HSV", action='store_true', help='Use HSV filtering.')
    parser.add_argument("--Gaussian-kernel-size", type=int, default=11, help='Gaussian kernel size.')
    parser.add_argument("--cluster-component", type=int, default=1, help='HSV components to cluster.')
    parser.add_argument("--colormap", type=int, default=16, help='Colormap, default=VIRIDIS (16).')
    parser.add_argument("-v", "--verbose", action='store_true', help='Print OpenCV and Matplotlib version.')
    parser.add_argument("--HDBSCAN", action='store_true', help='Use HDBSCAN instead of DBSCAN clustering algorithm.')
    parser.add_argument("--fill-contours", action='store_true', help='Fill contours before processing clustering algorithm.')
    args = parser.parse_args()

    input_image_path = args.input
    subsample_factor = args.subsample if args.subsample > 0 and args.subsample <= 10 else 1
    eps = args.eps
    min_samples = args.min_samples
    save = args.save
    no_roi = args.no_roi
    input_roi = args.roi
    hsv_min_range = args.HSV_min
    hsv_max_range = args.HSV_max
    use_HSV = args.HSV
    gaussian_kernel_size = args.Gaussian_kernel_size
    cluster_index = args.cluster_component
    colormap = args.colormap
    verbose = args.verbose
    use_HDBSCAN = args.HDBSCAN
    fill_contours = args.fill_contours

    print(f"Input image: {input_image_path}")
    print(f"Subsample factor: {subsample_factor}")
    print(f"DBSCAN eps: {eps}")
    print(f"DBSCAN min_samples: {min_samples}")
    print(f"save: {save}")
    print(f"No ROI?: {no_roi}")
    print(f"Input ROI: {input_roi}")
    print(f"Use HSV?: {use_HSV}")
    if use_HSV:
        print(f"HSV min range: {hsv_min_range}")
        print(f"HSV max range: {hsv_max_range}")
    print(f"Gaussian kernel size: {gaussian_kernel_size}x{gaussian_kernel_size}")
    print(f"Components to cluster: {cluster_index}")
    print(f"Colormap: {colormap}")
    print(f"Use HDBSCAN?: {use_HDBSCAN}")
    print(f"Fill contours?: {fill_contours}")

    if verbose:
        print()
        print('sklearn: {}'.format(sklearn.__version__))
        print('matplotlib: {}'.format(matplotlib.__version__))
        print(cv.getBuildInformation())

    if not Path(input_image_path).is_file():
        raise FileNotFoundError(f"Invalid input image: {input_image_path}")

    image_original = cv.imread(input_image_path)
    image_subsample = subsample(image_original, subsample_factor)
    print(f"image_original={image_original.shape}")
    print(f"image_subsample={image_subsample.shape}")

    if use_HSV:
        colorLower = hsv_min_range
        colorUpper = hsv_max_range
        print(f"colorLower={colorLower}")
        print(f"colorUpper={colorUpper}")

    if not no_roi:
        if input_roi is not None:
            roi = input_roi
        else:
            roi = cv.selectROI(image_subsample)
    else:
        roi = (0, 0, image_subsample.shape[1], image_subsample.shape[0])
    print(f"roi={roi}")

    # HSV
    # Compute dominant hue value into the ROI
    cropped_image = image_subsample[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    print(f"cropped_image={cropped_image.shape}")
    hsv, hue_masked, saturation_masked, _, inverse_mask = computeHSV(cropped_image)

    hue_component_flatten = hsv[:,:,0].reshape(-1, 1)
    sat_component_flatten = hsv[:,:,1].reshape(-1, 1)
    hue_sat_components = np.concatenate((hue_component_flatten, sat_component_flatten), axis=1)

    if use_HSV:
        blurred = cv.GaussianBlur(cropped_image, (gaussian_kernel_size, gaussian_kernel_size), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv, colorLower, colorUpper)
        element = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, element, iterations=2)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, element, iterations=2)
    else:
        mask = 255*np.ones((cropped_image.shape[0], cropped_image.shape[1]), dtype=np.uint8)
    # HSV

    # Contours
    cnts, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    biggest_contour = None
    max_area = 0
    for cnt in cnts:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            biggest_contour = cnt
    filled_mask = mask
    if biggest_contour is not None:
        filled_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv.drawContours(filled_mask, [biggest_contour], 0, 255, cv.FILLED)

    blurred = cv.GaussianBlur(cropped_image, (3, 3), 0)
    blurred_masked = cv.copyTo(blurred, filled_mask)
    cropped_image_gray = cv.cvtColor(blurred_masked, cv.COLOR_BGR2GRAY)
    sobel_x = cv.Sobel(cropped_image_gray, cv.CV_16SC1, 1, 0)
    sobel_y = cv.Sobel(cropped_image_gray, cv.CV_16SC1, 0, 1)
    edges_map_masked = cv.Canny(sobel_x, sobel_y, 100, 150)
    cnts, hierarchy = cv.findContours(edges_map_masked, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    img_cnts_gray = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
    img_cnts = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(cnts)):
        color = getLinear(i, 0, len(cnts)) # TODO: i or i+1?
        cv.drawContours(img_cnts_gray, cnts, i, color, 1, cv.LINE_8, hierarchy, 0)
    img_cnts = cv.applyColorMap(img_cnts_gray, cv.COLORMAP_VIRIDIS)

    img_cnts_upsample = upsample(img_cnts, subsample_factor)
    cv.putText(img_cnts_upsample, "Contours", (10,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    # Contours

    # Fill contours
    # TODO: warning about cv.RETR_LIST method --> no hierarchy <--> holes
    img_fill_cnts_bgr = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    print(f"cnts={len(cnts)}")
    for i in range(len(cnts)):
        cnts_mask = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
        cv.drawContours(cnts_mask, cnts, i, 255, cv.FILLED, cv.LINE_8, hierarchy, 0)
        mean_color = cv.mean(cropped_image, cnts_mask)
        cv.drawContours(img_fill_cnts_bgr, cnts, i, mean_color, cv.FILLED, cv.LINE_8, hierarchy, 0)
        # cv.imshow("cnts_mask", cnts_mask)
        # cv.waitKey(0)
    # Fill contours

    # DBSCAN
    if fill_contours:
        hsv, hue_masked, saturation_masked, _, inverse_mask = computeHSV(img_fill_cnts_bgr, mask)
    else:
        hsv, hue_masked, saturation_masked, _, inverse_mask = computeHSV(cropped_image, mask)
    hue_upsample = upsample(hsv[:,:,0], subsample_factor)
    saturation_upsample = upsample(hsv[:,:,1], subsample_factor)
    value_upsample = upsample(hsv[:,:,2], subsample_factor)
    cropped_image_upsample = upsample(cropped_image, subsample_factor)

    hue_component_flatten = hsv[:,:,0].reshape(-1, 1)
    sat_component_flatten = hsv[:,:,1].reshape(-1, 1)
    hue_sat_components = np.concatenate((hue_component_flatten, sat_component_flatten), axis=1)
    print(f"hsv={hsv.shape}")
    print(f"hue_component_flatten={hue_component_flatten.shape}")
    print(f"sat_component_flatten={sat_component_flatten.shape}")
    print(f"hue_sat_components={hue_sat_components.shape}")


    ### Grid visualization

    # Image gradient
    sobel_x_abs = cv.convertScaleAbs(sobel_x)
    sobel_y_abs = cv.convertScaleAbs(sobel_y)
    sobel_xy_8u = cv.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, gamma=0)
    sobel_xy_upsample_bgr = cv.cvtColor(upsample(sobel_xy_8u, subsample_factor), cv.COLOR_GRAY2BGR)
    cv.putText(sobel_xy_upsample_bgr, "Gradient", (10,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

    # Hue + Saturation
    hue_upsample_bgr = cv.cvtColor(hue_upsample, cv.COLOR_GRAY2BGR)
    cv.putText(hue_upsample_bgr, "Hue", (10,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    saturation_upsample_bgr = cv.cvtColor(saturation_upsample, cv.COLOR_GRAY2BGR)
    cv.putText(saturation_upsample_bgr, "Saturation", (10,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    value_upsample_bgr = cv.cvtColor(value_upsample, cv.COLOR_GRAY2BGR)
    cv.putText(value_upsample_bgr, "Value", (10,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))


    # Grid
    ngrid_h = 2
    ngrid_w = 4
    output_h = cropped_image_upsample.shape[0]
    output_w = cropped_image_upsample.shape[1]
    output = np.zeros((ngrid_h*output_h, ngrid_w*output_w, 3), dtype=np.uint8)
    output[:output_h, :output_w] = cropped_image_upsample
    output[:output_h, output_w:2*output_w] = sobel_xy_upsample_bgr
    output[output_h:, :output_w:] = value_upsample_bgr

    output[:output_h, 2*output_w:3*output_w] = hue_upsample_bgr
    output[output_h:, 2*output_w:3*output_w] = saturation_upsample_bgr

    # Contours
    output[:output_h, 3*output_w:] = img_cnts_upsample
    # Filled Contours
    img_fill_cnts_bgr_upsample = upsample(img_fill_cnts_bgr, subsample_factor)
    cv.putText(img_fill_cnts_bgr_upsample, "Filled Contours", (10,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
    output[output_h:2*output_h, 3*output_w:] = img_fill_cnts_bgr_upsample

    final_output_width = output.shape[1]
    final_output_height =  output.shape[0]

    labels = None
    final_output = None
    image_from_plot_bgr = None

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(hue_sat_components)
    # db = DBSCAN(eps=eps, min_samples=min_samples).fit(hue_component_flatten)
    # db = DBSCAN(eps=eps, min_samples=min_samples).fit(sat_component_flatten)
    i_coord, j_coord = np.mgrid[0:cropped_image.shape[0], 0:cropped_image.shape[1]]
    i_j_hue_components = np.concatenate((i_coord.reshape(-1,1), j_coord.reshape(-1,1), hue_component_flatten), axis=1)
    if use_HDBSCAN:
        # db = HDBSCAN(min_samples=min_samples).fit(hue_component)
        db = HDBSCAN(min_samples=min_samples).fit(i_j_hue_components)
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(i_j_hue_components)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # print(f"Nb clusters={n_clusters_} ; Nb noise points={n_noise_} ; len(labels)={len(labels)} ; min(labels)={np.min(labels)} ; max(labels)={np.max(labels)}")

    if n_noise_ > 0:
        labels = labels + 1 # to avoid having -1 label in the image
    labels_reshape = labels.reshape((cropped_image.shape[0], cropped_image.shape[1]))
    # print(f"labels_reshape={labels_reshape.shape}")

    n_labels = n_clusters_
    if n_noise_ > 0:
        n_labels += 1
    labels_visu = visuLabels(labels_reshape, n_labels, colormap) # n_clusters_-1
    labels_visu_upsample = upsample(labels_visu, subsample_factor)
    clustering_title = "HDBSCAN clusters" if use_HDBSCAN else "DBSCAN clusters"
    cv.putText(labels_visu_upsample, clustering_title, (10,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

    # Display labels
    output[output_h:, output_w:2*output_w] = labels_visu_upsample

    # Scatter plot
    fig, ax1 = plt.subplots(1, 1)
    ax1.clear()
    if labels is not None:
        ax1.scatter(hue_sat_components[:, 0], hue_sat_components[:, 1], c=labels)
        text_kwargs = dict(ha='center', va='center', fontsize=16, color='C1')
        ax1.text(190, 220, "DBSCAN\neps={} min_samples={}\nclusters: {}\nnoise: {}".format(eps, min_samples, n_clusters_, n_noise_), **text_kwargs)
    else:
        ax1.scatter(hue_sat_components[:, 0], hue_sat_components[:, 1])
    ax1.set_xlabel("Hue", fontsize=18)
    ax1.set_ylabel("Saturation", fontsize=18)
    ax1.set_xlim(0, 255)
    ax1.set_ylim(0, 255)
    ax1.grid()

    image_from_plot_bgr = imageFromPlot(fig, ax1, padding=1)
    print(f"image_from_plot_bgr={image_from_plot_bgr.shape} ; dtype={image_from_plot_bgr.dtype}")
    # print(f"output={output.shape} ; image_from_plot_bgr={image_from_plot_bgr.shape} ; final_output={final_output.shape}")

    # 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(i_coord, j_coord, hue_component_flatten, marker='o', c=labels)
    # ax.scatter(i_coord, j_coord, sat_component_flatten, marker='o')
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_zlabel('Hue')
    img_i_j_hue = imageFromPlot(fig, ax, padding=1)
    print(f"img_i_j_hue={img_i_j_hue.shape} ; dtype={img_i_j_hue.dtype}")

    # Merge image and plotting into a final output image
    print(f"output={output.shape} ; dtype={output.dtype}")
    final_output_width = max(output.shape[1], image_from_plot_bgr.shape[1] + img_i_j_hue.shape[1])
    final_output_height =  output.shape[0] + max(image_from_plot_bgr.shape[0], img_i_j_hue.shape[0])
    print(f"final_ouptut: {final_output_height}x{final_output_width}")

    final_output = np.zeros((final_output_height, final_output_width, 3), dtype=np.uint8)
    final_output[:output.shape[0], :output.shape[1]] = output
    final_output[output.shape[0]:output.shape[0]+image_from_plot_bgr.shape[0],
                 :image_from_plot_bgr.shape[1]] = image_from_plot_bgr
    final_output[output.shape[0]:output.shape[0]+img_i_j_hue.shape[0],
                 image_from_plot_bgr.shape[1]:image_from_plot_bgr.shape[1]+image_from_plot_bgr.shape[1]:] = img_i_j_hue
    cv.imshow("output", final_output)

    # # Histogram
    # fig, axs = plt.subplots(1, 2, sharex=True, tight_layout=True)
    # axs[0].hist(hue_component_flatten, bins=256)
    # axs[1].hist(sat_component_flatten, bins=256)

    # axs[0].set_xlim(left=0, right=255)
    # axs[1].set_xlim(left=0, right=255)
    # cv.imshow("3d plot", img_i_j_hue)

    if save:
        Path(save).mkdir(parents=True, exist_ok=True)
        current_date = datetime.today().strftime('%Y-%m-%d_%H_%M_%S')
        output_filename = f"images_{current_date}"
        output_filepath = str(Path(save, output_filename).with_suffix(".png"))
        cv.imwrite(output_filepath, final_output)

    cv.waitKey(0)

if __name__ == '__main__':
    main()
