import argparse
import os
import glob

DATA_DIRS = [
# "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-01.1\\",
# "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-04\\"
"/home/jay/Self-Driving-RC-Data/recorded-2017-06-01.01/",
"/home/jay/Self-Driving-RC-Data/recorded-2017-06-04/"
]
MODEL_H5 = os.path.abspath('/home/jay/Self-Driving-RC-Data/model.h5')

parser = argparse.ArgumentParser(description='Preprocess images')
parser.add_argument('model', type=str,
    default=MODEL_H5,
    help="Path to output model H5 file e.g. `/home/user/model.h5`.")
parser.add_argument('--data-dirs', type=str,
    default=DATA_DIRS,
    nargs='+',
    help="Path to directory that contains csv file and a directory of images.")
args = parser.parse_args()
print(glob.glob(os.path.join(args.data_dirs[1], '*.csv'))[0])
