import os
import numpy as np
from transformers import BertTokenizer
import datasets
import tarfile

# Step 1: Load and extract the BookCorpus dataset
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
            description="BookCorpus Dataset",
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://yknzhu.wixsite.com/mbweb",
            citation="",
        )

    def _split_generators(self, dl_manager):
        arch_path = '/path/to/bookcorpus.tar.bz2'  # Replace with your actual path
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

# Replace with your actual paths
archive_path = '/usr2/vishwaja/conditional-generation/bert/bookcorpus.tar.bz2'
extract_dir = '/usr2/vishwaja/conditional-generation/bert/extracted_files'

# Extract the dataset if not already extracted

# Collect files from the extracted directory
files = []
for root, _, filenames in os.walk(extract_dir):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        files.append((file_path, open(file_path, 'rb')))

# Create an instance of the BookCorpus class
book_corpus = Bookcorpus()

# Step 2: Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 3: Tokenize the text examples to obtain token IDs
token_ids_list = []
max_length = 128  # Maximum sequence length for BERT

# Define batch size based on your system's memory capacity
batch_size = 10000  # Adjust as needed

# Process the examples in batches
examples = book_corpus._generate_examples(files)
batch_texts = []
processed_examples = 0

for _id, example in examples:
    text = example['text']
    if text.strip():
        batch_texts.append(text)
        processed_examples += 1

    # Tokenize and save when batch is full
    if len(batch_texts) == batch_size:
        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )
        token_ids = inputs['input_ids']  # Numpy array of token IDs
        token_ids_list.append(token_ids)
        batch_texts = []
        print(f"Processed {processed_examples} examples.")
        if processed_examples >= 100000:
            break

# Tokenize any remaining texts
# if batch_texts:
#     inputs = tokenizer(
#         batch_texts,
#         add_special_tokens=True,
#         max_length=max_length,
#         truncation=True,
#         padding='max_length',
#         return_tensors='np'
#     )
#     token_ids = inputs['input_ids']
#     token_ids_list.append(token_ids)
#     print(f"Processed {processed_examples} examples.")

# Step 4: Save the token IDs into a .npy file
# Concatenate all token IDs into a single array
all_token_ids = np.concatenate(token_ids_list, axis=0)

# Save to .npy file
output_file = 'bookcorpus_token_ids.npy'
np.save(output_file, all_token_ids)
print(f"Token IDs saved to {output_file}")
