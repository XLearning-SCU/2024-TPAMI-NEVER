import json
import os

import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset

from dataset.utils import binary2img, pre_question


class vqa_dataset(Dataset):
    def __init__(
        self,
        ann_file,
        transform,
        vqa_root=None,
        vg_root=None,
        eos="[SEP]",
        split="train",
        max_ques_words=30,
        answer_list="",
    ):
        self.split = split
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

        print(f"{split}: {len(self.ann)}")
        print(self.image_tables.keys())
        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == "test":
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, "r"))

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
            if ann["dataset"] == "vqa":
                image_path = os.path.join(self.vqa_root, ann["image"])
            elif ann["dataset"] == "vg":
                image_path = os.path.join(self.vg_root, ann["image"])
            image = self.get_image(image_path=image_path)

        image = self.transform(image)

        if self.split == "test":
            question = pre_question(ann["question"], self.max_ques_words)
            question_id = ann["question_id"]
            return image, question, question_id

        elif self.split == "train":

            question = pre_question(ann["question"], self.max_ques_words)

            if ann["dataset"] == "vqa":

                answer_weight = {}
                for answer in ann["answer"]:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann["answer"])
                    else:
                        answer_weight[answer] = 1 / len(ann["answer"])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann["dataset"] == "vg":
                answers = [ann["answer"]]
                weights = [0.5]

            answers = [answer + self.eos for answer in answers]

            return image, question, answers, weights
