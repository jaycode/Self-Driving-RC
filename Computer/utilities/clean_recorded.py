# Clean recorded data i.e. ensuring csv and images contain the same data.
# Report any inconsistencies.

import glob
import csv
import os
import argparse

RECORDED_DIR = "/home/sku/recorded"
RECORDED_CSV = "/home/sku/recorded.csv"

def remove_rows(imgs, reader):
    pass

def remove_images(imgs, reader):
    pass

def main():
    rows = []
    with open(RECORDED_CSV, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)
    row_count = len(rows)
    imgs = glob.glob(os.path.join(RECORDED_DIR, '*.jpg'))
    img_count = len(imgs)

    parser = argparse.ArgumentParser(description='Clean recorded data')
    parser.add_argument('-r', type=str,
    help="Remove inconsistent data.")
    parser.add_argument('--dir', type=str,
        default=RECORDED_DIR,
        help="Path to recorded directory that contains images")
    parser.add_argument('--csv', type=str,
        default=RECORDED_CSV,
        help="Path to recorded csv file")

    print("Rows: {} Images: {}".format(row_count, img_count))
    remove_rows(imgs, reader)
    remove_images(imgs, reader)

if __name__ == "__main__":
    main()