import json
import os

import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset

from dataset.utils import binary2img, pre_caption


class ve_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, "r"))
        self.image_table = None
        if "arrow" in ann_file:
            self.image_table = pa.ipc.RecordBatchFileReader(
                pa.memory_map(ann_file[:-5], "r")
            ).read_all()

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.labels = {"entailment": 2, "neutral": 1, "contradiction": 0}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if self.image_table:
            image_index = ann["arrow_index"]
            image = binary2img(self.image_table["image"][image_index].as_py())
        else:
            image_path = os.path.join(self.image_root, "%s.jpg" % ann["image"])
            image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        sentence = pre_caption(ann["sentence"], self.max_words)

        return image, sentence, self.labels[ann["label"]]

