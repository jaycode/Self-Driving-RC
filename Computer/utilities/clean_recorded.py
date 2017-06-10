# Clean recorded data i.e. ensuring csv and images contain the same data.
# Report any inconsistencies.

import glob
import csv
import os
import argparse
import copy
import collections


RECORDED_DIR = "/home/sku/recorded"
RECORDED_CSV = "/home/sku/recorded.csv"
RECORDED_EDIT_CSV = "/home/sku/recorded.edit.csv"

# data with latency higher than this will be removed.
LATENCY_THRESHOLD = 0.5
REC_LATENCY_FIELD_ID = 3
FILENAME_FIELD_ID = 0

def remove_images(imgs, i_rows):
    images_to_remove = []
    rows = copy.copy(i_rows)
    total = 0
    print("Inconsistent images:")
    for img in imgs:
        name = os.path.basename(img)
        found = False
        for i, row in enumerate(rows):
            if row[FILENAME_FIELD_ID] == name:
                found = name
                rows.pop(i)
                break
        if not found:
            total += 1
            print(name)
            images_to_remove.append(img)
    if total == 0:
        print("None")
    return images_to_remove

def remove_rows(i_imgs, rows):
    rows_to_remove = []
    imgs = copy.copy(i_imgs)
    total = 0
    print("Inconsistent rows:")
    for row in rows:
        found = False
        for i, img in enumerate(imgs):
            name = os.path.basename(img)
            if name == row[FILENAME_FIELD_ID]:
                found = img
                imgs.pop(i)
                break
        if not found:
            if row[FILENAME_FIELD_ID] != 'filename': # do not remove header
                total += 1
                print(row[FILENAME_FIELD_ID])
                rows_to_remove.append(row)
    if total == 0:
        print("None")
    return rows_to_remove

def remove_hilat(imgs, rows):
    data_to_remove = []
    print("High latency data (>{})".format(LATENCY_THRESHOLD))
    for row in rows:
        try:
            if float(row[REC_LATENCY_FIELD_ID]) > LATENCY_THRESHOLD:
                print(row[FILENAME_FIELD_ID])
                data_to_remove.append(row[FILENAME_FIELD_ID])
        except:
            pass
    return data_to_remove

def main():
    parser = argparse.ArgumentParser(description='Clean recorded data')
    parser.add_argument('-r', action='store_true',
        default=False,
        help="When this flag is included, remove the images and rows for real.")
    parser.add_argument('--dir', type=str,
        default=RECORDED_DIR,
        help="Path to recorded directory that contains images")
    parser.add_argument('--csv', type=str,
        default=RECORDED_CSV,
        help="Path to recorded csv file")
    parser.add_argument('--edited-csv', type=str,
        default=RECORDED_EDIT_CSV,
        help="Path to edited recorded csv file")
    args = parser.parse_args()
    remove = False
    if args.r:
        remove = True
    recorded_dir = args.dir
    recorded_csv = args.csv
    recorded_edit_csv = args.edited_csv

    rows = []
    with open(recorded_csv, 'r') as csvfile, open(recorded_edit_csv, 'w') as out:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(row)

        # rows = sorted(rows)[0:6]
        row_count = len(rows)
        imgs = glob.glob(os.path.join(recorded_dir, '*.jpg'))
        # imgs = sorted(imgs)[0:5]
        img_count = len(imgs)


        rows_to_remove = remove_rows(imgs, rows)
        images_to_remove = remove_images(imgs, rows)
        print("Duplicate rows:")
        duplicate_rows = [item for item, count in collections.Counter(map(lambda r: r[0], rows)).items() if count > 1]

        for row in duplicate_rows:
            print(row)

        # Remove high latency data
        hilat_data = remove_hilat(imgs, rows)

        print("")
        print("Rows: {} Images: {}".format((row_count-1), img_count)) # account for header in row count.
        print("total duplicate rows:", len(duplicate_rows))
        # TODO: Remove duplicated rows when there is a scenario that requires so.

        print("total inconsistent rows:", len(rows_to_remove))
        print("total inconsistent images:", len(images_to_remove))
        print("Total high-latency data (>{}): {}".format(LATENCY_THRESHOLD, len(hilat_data)))

        if remove:
            total = 0
            writer = csv.writer(out)
            for final_row in [row for row in rows if (row not in rows_to_remove and row[FILENAME_FIELD_ID] not in hilat_data)]:
                writer.writerow(final_row)
                total += 1

            print("Saved {} rows into file {} (you need to manually replace the original csv)".format(total, recorded_edit_csv))

            total = 0
            for img in images_to_remove:
                total += 1
                os.remove(img)
            print("removed {} images".format(total))

            for img in hilat_data:
                total += 1
                path = os.path.join(recorded_dir, img)
                os.remove(path)
            print("removed {} images of high-latency data".format(total))

if __name__ == "__main__":
    main()