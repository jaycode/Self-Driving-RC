""" Prepare Polynet Data

Prepares images to create polynomials with for Polynet data.

Inputs:
- X: 310 x 240 px image, 3 channels.
Outputs:
- y: 3 coefficients for polynomial + 1 boolean for each line (totalling 8 outputs).


"""

import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)
from libraries.helpers import configuration, prepare_initial_transformation, preprocess_line_finding, load_from
from libraries.find_lines_sliding_windows import FindLinesSlidingWindows
from libraries.line_helpers import annotate_with_lines

import glob
import cv2
import numpy as np
import pdb
import argparse
import json
import pdb

config = configuration()

# Make sure the target shape is the same with the one in driver/main.py
# i.e. look for cams setup with variable CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT.
TARGET_WIDTH = config['target_width']
TARGET_HEIGHT = config['target_height']
TARGET_CROP = config['target_crop']
STEER_MIN = config['steer_min']
STEER_MAX = config['steer_max']
CHANNELS = config['channels']
NORMALIZE = config['normalize']


# There are multiple ways to find the polynomials that will have different coverages for each.

def to_label(fits):
    """ Extract information needed to create a label from a fits object.
    """
    if len(fits['left']):
        left_label = (True, *fits['left'][0]['poly'])
    else:
        # These zeroes may create problems later. Write custom loss
        # function when that happens.
        left_label = (False, 0, 0, 0)

    if len(fits['right']):
        right_label = (True, *fits['right'][0]['poly'])
    else:
        # These zeroes may create problems later. Write custom loss
        # function when that happens.
        right_label = (False, 0, 0, 0)
    return ((*left_label, *right_label))

def setup_polynomial_data(poly_config):
    """ Setup data from field "images".

    First poly object MUST be "default", and images in this poly object
    must be substracted from the images from other poly objects.

    For default poly object, field "images" will contain a list of globs.

    For other than the default poly object, for each item in field "images":
      - If it is a `.txt` file, load the filenames there. Store image paths
        from the "default" poly with the same filename.
      - Otherwise, treat as glob. No need to make sure each filename is contained in "default" poly.

    Returns:
      - Count of all rows.
      - Polynomial function names.
      - List of image paths for each polynomial function.
    """
    default_image_paths = []
    poly_image_paths = []
    polynames = ["default"]
    default_poly = poly_config["default"]
    row_count = 0

    # Adding image paths from default poly.
    for img_paths_field in default_poly["images"]:
        default_image_paths += glob.glob(img_paths_field)

    for polyname in poly_config:
        if polyname != "default":
            polynames.append(polyname)
            poly = poly_config[polyname]
            for image_paths_field in poly["images"]:
                # If a txt file, load the data from that file,
                # otherwise use glob.
                if image_paths_field[-3:] == "txt":
                    filenames = load_from(image_paths_field)
                    image_paths = [i for i in default_image_paths if os.path.basename(i) in filenames]
                else:
                    image_paths = glob.glob(image_paths_field)

                default_image_paths = [i for i in default_image_paths if i not in image_paths]
                print("{} - number of images: {}".format(polyname, len(image_paths)))
                poly_image_paths.append(image_paths)
                row_count += len(image_paths)
    
    poly_image_paths.insert(0, default_image_paths)
    print("number of default images:",len(default_image_paths))
    row_count += len(default_image_paths)
    return (row_count, polynames, poly_image_paths)

def add_function(flist, poly):
    def default_function(img_raw, height, width, crop, M):
        finder = FindLinesSlidingWindows(**poly["finder"])
        if "preprocess" in poly:
            preprocessed = preprocess_line_finding(img_raw, M, **poly["preprocess"])
        else:
            preprocessed = preprocess_line_finding(img_raw, M)
        fits = finder.process(preprocessed)
        return fits
    flist.append(default_function)

def setup_polynomial_functions(poly_config):
    """ Setting up polynomial functions from config.
    """
    poly_functions = []
    default_poly = poly_config["default"]
    add_function(poly_functions, default_poly)
    for polyname in poly_config:
        if polyname != "default":
            add_function(poly_functions, poly_config[polyname])
    return poly_functions


def main():
    parser = argparse.ArgumentParser(description='Clean recorded data')
    parser.add_argument('--poly-config', type=str,
        help="Config JSON file")
    parser.add_argument('--results-dir', type=str,
        help="Path to the images + predicted lane lines for debugging.")
    parser.add_argument('--results-csv', type=str,
        help="Path to the csv file containing the polynomials and other information.")
    parser.add_argument('--calibration-file', type=str,
        help="Path to calibration file e.g. `/home/user/cal-elp.p`.")
    args = parser.parse_args()
    poly_config_path = args.poly_config
    results_dir = args.results_dir
    results_csv = args.results_csv
    calibration_file = args.calibration_file

    with open(poly_config_path) as json_data:
        poly_config = json.load(json_data)

    mtx, dist, M, Minv = prepare_initial_transformation(
        calibration_file, TARGET_HEIGHT, TARGET_WIDTH)

    # Poly config objects, starting from "default"
    if "default" not in poly_config:
        raise ValueError("Poly config file must contain a \"default\" poly object.")

    row_count, polynames, poly_image_paths = setup_polynomial_data(poly_config)
    functions = setup_polynomial_functions(poly_config)

    print("Poly config file: {}".format(poly_config_path))
    print("Saving images to {}".format(results_dir))
    print("and data to {}".format(results_csv))
    counter = 0
    c = 0

    fd = open(results_csv, 'w')
    head = "left_hasline,left_coef1,left_coef2,left_coef3,right_hasline,right_coef1,right_coef2,right_coef3,imgpath\n"
    fd.write(head)

    for i, image_paths in enumerate(poly_image_paths):
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is not None:
                # print("image_path:", image_path)
                # print("img is ", img)
                fits = functions[i](img, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CROP, M)
                
                if len(fits['left']) and len(fits['right']):
                    path = os.path.join(results_dir, 'both')
                elif len(fits['left']):
                    path = os.path.join(results_dir, 'left')
                elif len(fits['right']):
                    path = os.path.join(results_dir, 'right')
                else:
                    path = os.path.join(results_dir, 'none')
                os.makedirs(path, exist_ok=True)

                # Annotate and save the image.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                warped = cv2.warpPerspective(img, M, (TARGET_WIDTH, TARGET_HEIGHT))

                annotated = annotate_with_lines(warped, fits)
                annotated = cv2.warpPerspective(annotated, Minv, (TARGET_WIDTH, TARGET_HEIGHT))
                annotated = cv2.putText(annotated, polynames[i], (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                filename = os.path.basename(image_path)
                imgpath = os.path.join(path, filename)
                cv2.imwrite(imgpath, annotated)

                # Add the row to csv file.
                row = "{},{},{},{},{},{},{},{},{}\n".format(*to_label(fits), imgpath)
                fd.write(row)

            if c%(max(1,int(row_count/40))) == 0:
                counter+=1
                sys.stdout.write("\r{0}".format("="*counter))
            c+=1

    print("\n")
    fd.close()

if __name__ == "__main__":
    main()