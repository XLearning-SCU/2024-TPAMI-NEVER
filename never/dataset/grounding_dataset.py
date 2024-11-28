import json
import os

import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset

from dataset.utils import binary2img, pre_caption


class grounding_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30, mode="train"):
        self.ann = []
        self.image_tables = {}
        start_index = 0
        for f in ann_file:
            ann = json.load(open(f, "r"))
            self.ann += ann
            if "arrow" in f:
                # xxx.arrow.json: images stored in xxx.arrow
                # if we have multiple ann_files, we need store the index_range and corresponding arrow table
                end_index = start_index + len(ann)
                self.image_tables[
                    (start_index, end_index)
                ] = pa.ipc.RecordBatchFileReader(pa.memory_map(f[:-5], "r")).read_all()
                start_index = end_index

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.mode = mode

        if self.mode == "train":
            self.img_ids = {}
            n = 0
            for ann in self.ann:
                img_id = ann["image"].split("/")[-1]
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1

    def __len__(self):
        return len(self.ann)

    def get_image(self, image_path=None, image_index=None, table_key=None):
        assert image_path is not None or image_index is not None
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        if image_index is not None:
            image = binary2img(
                self.image_tables[table_key]["image"][image_index].as_py()
            )
        return image

    def __getitem__(self, index):

        ann = self.ann[index]

        if self.image_tables:
            for (start, end) in self.image_tables.keys():
                if start <= index < end:
                    image = self.get_image(
                        image_index=ann["arrow_index"], table_key=(start, end)
                    )
                    break
        else:
            image_path = os.path.join(self.image_root, ann["image"])
            image = self.get_image(image_path=image_path)

        image = self.transform(image)

        caption = pre_caption(ann["text"], self.max_words)

        if self.mode == "train":
            img_id = ann["image"].split("/")[-1]

            return image, caption, self.img_ids[img_id]
        else:
            return image, caption, ann["ref_id"]

