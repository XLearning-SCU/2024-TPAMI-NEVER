import io
import json
import os
import random

import pyarrow as pa
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import binary2img, pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        self.image_tables = {}
        start_index = 0
        for f in ann_file:
            ann = json.load(open(f, "r"))
            self.ann += ann
            if "arrow" in f:
                # xxx.arrow.json: images stored in xxx.arrow
                # if we have multiple ann_files, we need save the index_range and corresponding arrow table
                end_index = start_index + len(ann)
                self.image_tables[
                    (start_index, end_index)
                ] = pa.ipc.RecordBatchFileReader(pa.memory_map(f[:-5], "r")).read_all()
                start_index = end_index

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def get_image(self, image_path=None, image_index=None, table_key=None):
        assert image_path is not None or image_index is not None
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        if image_index is not None:
            image = binary2img(
                self.image_tables[table_key]["image"][image_index].as_py()
            )
        return image

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        if self.image_tables:
            for (start, end) in self.image_tables.keys():
                if start <= index < end:
                    image = self.get_image(
                        image_index=ann["arrow_index"], table_key=(start, end)
                    )
        else:
            image = self.get_image(
                image_path=os.path.join(self.image_root, ann["image"])
            )

        image1 = self.transform(image)
        image2 = self.transform(image)

        caption = pre_caption(ann["caption"], self.max_words)

        return image1, image2, caption, self.img_ids[ann["image_id"]]


class re_eval_dataset(Dataset):
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

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def get_image(self, image_path=None, image_index=None):
        assert image_path is not None or image_index is not None
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        if image_index is not None:
            image = binary2img(self.image_table["image"][image_index].as_py())
        return image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        ann = self.ann[index]
        if self.image_table is not None:
            image = self.get_image(image_index=ann["arrow_index"])
        else:
            image = self.get_image(
                image_path=os.path.join(self.image_root, ann["image"])
            )
        image = self.transform(image)

        return image, index


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, "r"))
        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        if type(ann["caption"]) == list:
            caption = pre_caption(random.choice(ann["caption"]), self.max_words)
        else:
            caption = pre_caption(ann["caption"], self.max_words)

        image = Image.open(ann["image"]).convert("RGB")
        image1 = self.transform(image)
        image2 = self.transform(image)

        return image1, image2, caption


class pretrain_dataset_arrow(Dataset):
    def __init__(self, ann_file, transform, max_words=30):

        tables = [
            pa.ipc.RecordBatchFileReader(pa.memory_map(file_path, "r")).read_all()
            for file_path in ann_file
            if os.path.isfile(file_path)
        ]
        self.table = pa.concat_tables(tables, promote=True)

        # caption
        remove_duplicate = False
        self.all_texts = self.table["caption"].to_pandas().tolist()
        self.all_texts = (
            [list(set(texts)) for texts in self.all_texts]
            if remove_duplicate
            else self.all_texts
        )

        # caption to image
        self.index_mapper = dict()
        j = 0
        for i, texts in enumerate(self.all_texts):
            # captions > image
            for _j in range(len(texts)):
                self.index_mapper[j] = (i, _j)
                j += 1

        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        # the size of all texts
        return len(self.index_mapper)

    def __getitem__(self, index):
        get_data = False
        while not get_data:
            # in case file error
            try:
                image_index, caption_index = self.index_mapper[index]
                # caption
                caption = pre_caption(
                    self.all_texts[image_index][caption_index], self.max_words
                )

                # image
                image_bytes = io.BytesIO(self.table["image"][image_index].as_py())
                image_bytes.seek(0)
                image = Image.open(image_bytes).convert("RGB")

                # augmentation
                image1 = self.transform(image)
                image2 = self.transform(image)

                get_data = True
            except Exception as e:
                index = random.randint(0, len(self.index_mapper) - 1)

        return image1, image2, caption

