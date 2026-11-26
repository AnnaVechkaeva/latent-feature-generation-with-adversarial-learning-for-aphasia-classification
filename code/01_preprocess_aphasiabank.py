# Step 1 - Preprocessing (paper Sec. 3.2)
# Reads AphasiaBank CHAT transcripts (.cha), keeps only participant (*PAR:) lines,
# removes CHAT markup, and reads the WAB aphasia type from the @ID line.
# Writes one cleaned .clean file per subject plus an id -> label file.
#
# Expects the raw transcripts in ./aphasia/English_scripts/ (output folders must exist).
# Controls were processed the same way by pointing data_path at the control transcripts.
#
# Original file name: data_preprocessing.py (2019). Code below is unchanged.

import os
import re


def split_data(data, length):
    for i in range(0, len(data), length):
        return data[i:i + length]

class Subject:
    def __init__(self):
        self.subj_id = None
        self.raw_text_lines = None
        self.clean_text = []
        self.aph_type = None
        self.path = None
        self.datapoints = []
        self.length = 1

    def _get_datapoints(self):
        text = self.clean_text
        length = self.length
        for i in range(0, len(text), length):
            self.datapoints.append(text[i:i + length])

    def _open_file(self):
        with open(self.path) as file:
            raw_text = file.read()
            self.raw_text_lines = raw_text.split('\n')

    def _clean(self):
        for line in self.raw_text_lines:
            if "*PAR:" in line:
                self.clean_text += line.split('\t')[1] +'\n'

    def _clean_everything(self):
        lines = []
        for line in self.raw_text_lines:
            if "*PAR:" in line:
                clean_line = line.split('\t')[1]
                clean_line = re.sub('\+…', ' ', clean_line)
                clean_line = re.sub('\+…\?', ' ', clean_line)
                clean_line = re.sub('\+/\.', ' ', clean_line)
                clean_line = re.sub('\+/\?', ' ', clean_line)
                clean_line = re.sub('\+//\.', ' ', clean_line)
                clean_line = re.sub('\+//\?', ' ', clean_line)
                clean_line = re.sub('\+\,', ' ', clean_line)
                clean_line = re.sub('\+\+', ' ', clean_line)
                clean_line = re.sub('\+\"\/\.', ' ', clean_line)
                clean_line = re.sub('\+"', ' ', clean_line)
                clean_line = re.sub('\+"\.', ' ', clean_line)
                clean_line = re.sub('\+\<', ' ', clean_line)
                clean_line = re.sub('&=(\S)+', '', clean_line)
                clean_line = re.sub('&-', '', clean_line)
                clean_line = re.sub('\[.+?\]', '', clean_line)
                clean_line = re.sub('&..? ', '', clean_line)
                clean_line = re.sub('‡', '', clean_line)
                clean_line = re.sub('\+', '', clean_line)
                clean_line = re.sub('(\(|\))', '', clean_line)
                clean_line = re.sub('(\<|\>)', '', clean_line)
                clean_line = re.sub('[0-9]{4,6}_[0-9]{4,6}', '', clean_line)
                clean_line = re.sub('_', ' ', clean_line)
                clean_line = re.sub('\:', '', clean_line)
                clean_line = re.sub('\.\.? ', '', clean_line)
                clean_line = re.sub('\.\. ', '', clean_line)
                clean_line = re.sub('  ', ' ', clean_line)
                clean_line = re.sub('^ ', '', clean_line)
                clean_line = re.sub('', '', clean_line)
                if len(clean_line) > 1:
                    lines.append(clean_line)
                    #self.clean_text += clean_line + '\n'
            self.clean_text = lines

    def _get_label(self):
        for line in self.raw_text_lines:
            if "@ID:" in line and "PAR" in line:
                content = line.split(':')[1]
                aphasia_type = content.split('|')[5]
                self.aph_type = aphasia_type
                break

    def _get_id(self):
        pid = re.search("\./aphasia/English_scripts/.*?/(.*?)\.cha", self.path)
        self.subj_id = pid.group(1)

    def set_up(self):
        self._open_file()
        self._get_id()
        self._clean_everything()
        self._get_label()
        self._get_datapoints()



class DataSet:
    def __init__(self):
        self.name = None
        self.data_path = None
        self.files_paths = []
        self.subjects = []
        self.datapoints = []

    def _find_paths(self):
        for root, dirs, files in os.walk(self.data_path, topdown=False):
            for name in files:
                if name != ".DS_Store" and name != "0metadata.cdc":
                    self.files_paths.append(os.path.join(root, name))

    def _get_subjects(self):
        for path in self.files_paths:
            subj = Subject()
            subj.path = path
            subj.set_up()
            self.subjects.append(subj)

    def build_data_set(self):
        self._find_paths()
        self._get_subjects()

    def write_data(self, output_dir):
        with open("./" + output_dir + "/" + "labels_file.txt", 'w', encoding = "utf-8") as labels_file:
            for subj in self.subjects:
                with open("./" + output_dir + "/" + subj.subj_id + ".clean", 'w', encoding="utf-8") as file:
                    file.write('\n'.join(subj.clean_text))
                    labels_file.write(subj.subj_id + '\t' + subj.aph_type + '\n')

    def write_datapoints(self, output_dir, treshold = None):
        length = 0
        with open("./" + output_dir + "/" + "datapoints.txt", 'w', encoding = "utf-8") as file:
            for subj in self.subjects:
                dp = subj.datapoints
                for dpoint in dp:
                    #print([dpoint, subj.subj_id, subj.aph_type])
                    dpoint = ' '.join(dpoint)
                    if len(dpoint) > length:
                        length = len(dpoint)
                        print(length)
                    if treshold is not None:
                        if len(dpoint.split()) > treshold:
                            file.write(subj.subj_id + '\t' + subj.aph_type + '\t' + dpoint + '\n')
                    else:
                        file.write(subj.subj_id + '\t' + subj.aph_type + '\t' + dpoint + '\n')

    def write_labels(self, output_name):
        with open(output_name, 'w', encoding="utf-8") as file:
            for subj in self.subjects:
                file.write(subj.subj_id + '\t' + subj.aph_type + '\n')


if __name__ == "__main__":
    aph_data = DataSet()
    aph_data.name = "Aphasia"
    aph_data.data_path = "./aphasia/English_scripts/"
    aph_data.build_data_set()
    aph_data.write_data("clean1")
    aph_data.write_labels("id_label_aph.txt")
    aph_data.write_datapoints("datapoints1_5", treshold = 5)
    for i in aph_data.subjects[:5]:
        print(i.aph_type)