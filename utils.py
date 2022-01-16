# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
csv.field_size_limit(100000000)
import glob
import json
import logging
import os
from typing import List
import tqdm
import secrets

from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, word, contexts, questions, endings, predicate_position, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            questions: list of str. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.word = word
        self.contexts = contexts
        self.questions = questions
        self.endings = endings
        self.n_choice = len(endings)
        self.predicate_position = predicate_position
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, predicate_position, n_choice, mlm_features, label, external_features=None):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.mlm_features = mlm_features
        self.predicate_position = predicate_position
        self.n_choice = n_choice
        self.external_features = external_features
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class ArtifactFunctionProcessor(DataProcessor):
    """Processor for the Frame data set."""
    def __init__(self, encode_type):
        self.encode_type = encode_type
        assert self.encode_type in {'lu_fn', 'ludef', 'fn', 'fndef', 'ludef_fn', 'ludef_fndef'}


    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        if data_dir.endswith('csv'):
            return self._create_examples(self._read_csv(data_dir), "test")
        else:
            return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        raise ValueError('No labels needed!')

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")
        # 0 artifact, 1 artifact definition, 2 label
        frames, defs = zip(*[x.split('\t') for x in open('frame_defs.tsv').read().splitlines()[1:]])
        if self.encode_type == 'fndef':
            examples = [
                InputExample(
                    example_id=secrets.token_hex(nbytes=16),
                    word=line[0],
                    contexts=[line[0]+': '+line[1]] * len(frames), # (context, question+ending) * n_choice
                    questions=frames,
                    endings=defs,
                    predicate_position=0,
                    label=line[2] if len(line)>2 else None,
                )
                for line in lines[1:]  # we skip the line with the column names
            ]
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    model_name_or_path: str,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:

    external_features = dict()
    def get_mlm_features(word):
        from pattern.en import referenced
        art_word = referenced(word, 'indefinite')
        if 'uncased' in model_name_or_path: art_word_cap = art_word
        else: art_word_cap = art_word.capitalize()
        patterns = [
                f'{art_word_cap} can be used to [MASK]',
                f'I used {art_word} to [MASK]',
                f'{art_word_cap} can be used for [MASK]',
                f'I used {art_word} for [MASK]',
                f'The purpose of {art_word} is to [MASK]',
                f'If I had {art_word}, I could [MASK]',
                ]
        inputs = tokenizer.batch_encode_plus(patterns, max_length=20, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        mask_ids = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        result = (input_ids, attention_mask, mask_ids)
        return result


    features = []
    sequence_cropping_count = 0
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        predicate_positions = []
        for ending_idx, (context, question, ending) in enumerate(zip(example.contexts, example.questions, example.endings)):
            text_a = context
            text_b = question + ": " + ending
            text_a = text_a.lower()
            text_b = text_b.lower()
            # In case text_b is too long
            text_b = ' '.join(text_b.strip().split()[:100])

            inputs = tokenizer.encode_plus(
                text_a, text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
                return_overflowing_tokens=True,
            )

            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                predicate_positions += [0]
                sequence_cropping_count += 1
            else:
                predicate_ids = tokenizer.encode(text_a.split()[int(example.predicate_position)], add_special_tokens=True)
                predicate_positions += [inputs['input_ids'].index(predicate_ids[1])]
                # print('\nInput too long that predicate is not/mistakenly found! Increase the max_length!')
            assert predicate_positions[-1] < inputs['input_ids'].index(tokenizer.sep_token_id)
            predicate_positions_nonzero = [x for x in predicate_positions if x != 0]
            assert all(x == predicate_positions_nonzero[0] for x in predicate_positions_nonzero)
            
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! You are cropping tokens!"
                    "You need to try to use a bigger max seq length!"
                )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = int(example.label) if example.label is not None else None

        if ex_index < 2:
            logger.info("*** Example ***")
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(
                InputFeatures(
                    example_id=example.example_id,
                    choices_features=choices_features,
                    predicate_position=predicate_positions[0]
                    if all(x==predicate_positions[0] for x in predicate_positions) else 0,
                    n_choice=example.n_choice,
                    mlm_features=get_mlm_features(example.word),
                    external_features=external_features[example.word]
                    if example.word in external_features else [0]*len(choices_features),
                    label=label,)
                )
    print('Sequence cropping:', sequence_cropping_count)

    return features


processors = {
        "artifact_function": ArtifactFunctionProcessor,
        }

