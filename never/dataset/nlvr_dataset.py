import json
import os

import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset

from dataset.utils import binary2img, pre_caption


class nlvr_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):
        self.ann = json.load(open(ann_file, "r"))
        self.image_table = None
        if "arrow" in ann_file:
            self.image_table = pa.ipc.RecordBatchFileReader(
                pa.memory_map(ann_file[:-5], "r")
            ).read_all()
        self.transform = transform
        self.image_root = image_root
        self.max_words = 30

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if self.image_table:
            image_index0 = ann["arrow_index"][0]
            image0 = binary2img(self.image_table["image"][image_index0].as_py())

            image_index1 = ann["arrow_index"][1]
            image1 = binary2img(self.image_table["image"][image_index1].as_py())
        else:
            image0_path = os.path.join(self.image_root, ann["images"][0])
            image0 = Image.open(image0_path).convert("RGB")

            image1_path = os.path.join(self.image_root, ann["images"][1])
            image1 = Image.open(image1_path).convert("RGB")

        image0 = self.transform(image0)
        image1 = self.transform(image1)

        sentence = pre_caption(ann["sentence"], self.max_words)

        if ann["label"] == "True":
            label = 1
        else:
            label = 0

        return image0, image1, sentence, label
