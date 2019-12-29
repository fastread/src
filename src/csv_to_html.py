"""
Converts the CSV files located at 'SOURCE'
to HTML files.

Each row of CSV become a HTML file.
Every 1000 HTML files are grouped in
separate directories.

The HTML file name is in format: TopicID-CSVrowNumber
"""

import csv
import os

SOURCE = "../workspace/data"
DEST = "../workspace/cormarck-data"

if not os.path.exists(DEST):
    os.mkdir(DEST)

csv_files = [os.path.join(SOURCE, f) for f in os.listdir(SOURCE) if f.split(".")[-1] == "csv"]
print(csv_files)

files_in_dir = 0
dir_index = 0

for csv_file_name in csv_files:
    topic = os.path.split(csv_file_name)[1].split(".")[0]

    csv_file = open(csv_file_name, "r")
    reader = csv.reader(csv_file)

    row_number = 0
    for row in reader:

        # header is ignored
        if row_number == 0:
            row_number += 1
            continue

        # creates a new directory
        if files_in_dir == 0:
            dir_name = os.path.join(DEST, str(dir_index).zfill(5))

            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

        # to change directory
        if files_in_dir == 999:
            files_in_dir = -1
            dir_index += 1


        # writes to html file
        with open(html_file_name, "w") as html_file:
            html_file.write("<TEXT>&#2;")
            html_file.write("<DATELINE>" + row[2] + "</DATELINE>")
            html_file.write("<TITLE>" + row[0] + "</TITLE>")
            html_file.write("<BODY>" + row[1])
            html_file.write("&#3;</BODY></TEXT>")

        files_in_dir += 1
        row_number += 1

    csv_file.close()
