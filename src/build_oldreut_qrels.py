import os

SOURCE_SUBTOPICS_FILE = "DTA_2017/training/qrels/qrel_abs_train"
DEST_SUBTOPICS_FILE = "../oldreut/qrels.list"

SOURCE_TOPIC_DIR = "DTA_2017/training/topics_train"
DEST_TOPIC_FILE = "../oldreut/topic.stemming"

MAP_FILE = "../oldreut/html-pid-mapping"

if not os.path.exists("../oldreut"):
    os.mkdir("../oldreut")

topic_map = {}

# Builds the file with topic names
topics = [os.path.join(SOURCE_TOPIC_DIR, f) for f in os.listdir(SOURCE_TOPIC_DIR)]
topic_files = [topic for topic in topics if os.path.isfile(topic)]

topic_number = 0
dst_topic_file = open(DEST_TOPIC_FILE, "w")

for topic_file_name in topic_files:
    with open(topic_file_name, "r") as topic_file:
        for i, line in enumerate(topic_file):
            # the title is at 3rd line in every file
            if i == 2:
                topic_title = line.split(": ")[1]
                dst_topic_file.write("tr" + str(topic_number) + ":" + topic_title)
                topic_map[topic_file_name.split("/")[-1].split(".")[0]] = "tr" + str(topic_number)

                break

    topic_number += 1

dst_topic_file.close()

# Builds the file with positives files only
dst_subtopics_file = open(DEST_SUBTOPICS_FILE, "w")

with open(MAP_FILE, "r") as map_file:

    with open(SOURCE_SUBTOPICS_FILE, "r") as subtopics_file:

        for line in subtopics_file.readlines():
            line_info = line.split()

            if line_info[3] == "1":

                html_file_number = None

                for map_line in map_file:
                    if map_line.split()[0] != line_info[0]:
                        continue

                    else:
                        if map_line.split()[1] != line_info[2]:
                            continue

                        else:
                            html_file_number = map_line.split()[2]
                            break

                print(html_file_number)
                dst_subtopics_file.write(
                        topic_map[line_info[0]] + " " +
                        line_info[1] + " " +
                        str(html_file_number) + " " +
                        line_info[3] + "\n"
                    )

dst_subtopics_file.close()
