# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""The BookCorpus dataset."""

import datasets


_DESCRIPTION = """\
Books are a rich source of both fine-grained information, how a character, \
an object or a scene looks like, as well as high-level semantics, what \
someone is thinking, feeling and how these states evolve through a story.\
This work aims to align books to their movie releases in order to provide\
rich descriptive explanations for visual content that go semantically far\
beyond the captions available in current datasets. \
"""

_CITATION = """\
@InProceedings{Zhu_2015_ICCV,
    title = {Aligning Books and Movies: Towards Story-Like Visual Explanations by Watching Movies and Reading Books},
    author = {Zhu, Yukun and Kiros, Ryan and Zemel, Rich and Salakhutdinov, Ruslan and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {December},
    year = {2015}
}
"""

URL = "https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2"


class BookcorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for BookCorpus."""

    def __init__(self, **kwargs):
        """BuilderConfig for BookCorpus.
        Args:
        **kwargs: keyword arguments forwarded to super.
        """
        super(BookcorpusConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class Bookcorpus(datasets.GeneratorBasedBuilder):
    """BookCorpus dataset."""

    BUILDER_CONFIGS = [
        BookcorpusConfig(
            name="plain_text",
            description="Plain text",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://yknzhu.wixsite.com/mbweb",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, archive):
        for _, ex in self._generate_examples(archive):
            yield ex["text"]

    def _split_generators(self, dl_manager):
        arch_path = dl_manager.download(URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"files": dl_manager.iter_archive(arch_path)}
            ),
        ]

    def _generate_examples(self, files):
        _id = 0
        for path, file in files:
            for line in file:
                yield _id, {"text": line.decode("utf-8").strip()}
                _id += 1

    # Your class implementation here

dl_manager = datasets.DownloadManager()

# Download the archive
archive_path = '/usr2/vishwaja/conditional-generation/bert/bookcorpus.tar.bz2'
extract_dir = '/usr2/vishwaja/conditional-generation/bert/extracted_files'

# arch_path = dl_manager.download(URL)
# Create an instance of the class
book_corpus = Bookcorpus()
import tarfile
with tarfile.open(archive_path, 'r:bz2') as archive:
    archive.extractall(path=extract_dir)

import os
files = []
for root, _, filenames in os.walk(extract_dir):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        files.append((file_path, open(file_path, 'rb')))




# Generate examples
examples = book_corpus._generate_examples(files)

# Iterate over the examples and print them
i = 0
for _id, example in examples:
    print(f"ID: {_id}, Text: {example['text']}")
    sentence = example['text']
    inputs = tokenizer(sentence, return_tensors="pt")
    i+=1
    if i == 5:
        break