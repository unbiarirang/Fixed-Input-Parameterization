from enum import Enum
import math
import json
from collections import defaultdict
import random
from dataclasses import dataclass
from typing import Any, Optional, Union
from datetime import datetime
from datetime import date
from pytz import timezone
import time

import numpy as np

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase



class Method(str, Enum):
    lm_finetune = "lm-finetune"
    random_matrix = "random-matrix"
    random_question = "random-question"
    electra_style = "electra-style"
    input_generation = "input-generation"
    hybrid = "hybrid"
    evaluate_only = "evaluate-only"


class Dataset(str, Enum):
    squad = "squad"
    personachat = "personachat"


class Optimizer(str, Enum):
    adam = "adam"
    adamw = "adamw"


def validate_arguments(method, dataset, valid_path):
    pass


# Copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(
    input_ids: np.array, pad_token_id: int, decoder_start_token_id: int
) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


def parse_valid_file(path):
    json_lines = []
    with open(path) as valid_file:
        for line in valid_file:
            json_lines.append(json.loads(line))

    data = defaultdict(list)
    for datum in json_lines:
        pid, _ = datum["id"].split("-")
        data[pid].append(datum)

    return data


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        labels_attention_mask = [feature["labels_attention_mask"] for feature in features] if "labels_attention_mask" in features[0].keys() else None
        if labels_attention_mask is not None:
            assert labels is not None
            
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["labels_attention_mask"]))
                if isinstance(feature["labels_attention_mask"], list):
                    feature["labels_attention_mask"] = (
                        feature["labels_attention_mask"] + remainder if padding_side == "right" else remainder + feature["labels_attention_mask"]
                    )
                elif padding_side == "right":
                    feature["labels_attention_mask"] = np.concatenate([feature["labels_attention_mask"], remainder]).astype(np.int64)
                else:
                    feature["labels_attention_mask"] = np.concatenate([remainder, feature["labels_attention_mask"]]).astype(np.int64)

                
        teacher_input_ids = [feature["teacher_input_ids"] for feature in features] if "teacher_input_ids" in features[0].keys() else None
        if teacher_input_ids is not None:
            teacher_input_ids_length = max(len(l) for l in teacher_input_ids)
            if self.pad_to_multiple_of is not None:
                teacher_input_ids_length = (
                    (teacher_input_ids_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (teacher_input_ids_length - len(feature["teacher_input_ids"]))
                if isinstance(feature["teacher_input_ids"], list):
                    feature["teacher_input_ids"] = (
                        feature["teacher_input_ids"] + remainder if padding_side == "right" else remainder + feature["teacher_input_ids"]
                    )
                elif padding_side == "right":
                    feature["teacher_input_ids"] = np.concatenate([feature["teacher_input_ids"], remainder]).astype(np.int64)
                else:
                    feature["teacher_input_ids"] = np.concatenate([remainder, feature["teacher_input_ids"]]).astype(np.int64)

        teacher_attention_mask = [feature["teacher_attention_mask"] for feature in features] if "teacher_attention_mask" in features[0].keys() else None
        if teacher_attention_mask is not None:
            teacher_attention_mask_length = max(len(l) for l in teacher_attention_mask)
            if self.pad_to_multiple_of is not None:
                teacher_attention_mask_length = (
                    (teacher_attention_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (teacher_attention_mask_length - len(feature["teacher_attention_mask"]))
                if isinstance(feature["teacher_attention_mask"], list):
                    feature["teacher_attention_mask"] = (
                        feature["teacher_attention_mask"] + remainder if padding_side == "right" else remainder + feature["teacher_attention_mask"]
                    )
                elif padding_side == "right":
                    feature["teacher_attention_mask"] = np.concatenate([feature["teacher_attention_mask"], remainder]).astype(np.int64)
                else:
                    feature["teacher_attention_mask"] = np.concatenate([remainder, feature["teacher_attention_mask"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


SENTINELS = [f"<extra_id_{i}>" for i in range(100)]
MAX_SPAN_LENGTH = 3
MIN_SPAN_LENGTH = 3


def generate_masked_sample(text, noise_density=0.15):
    mask = span_corruption_mask(text, noise_density)
    source_text = noise_span_to_unique_sentinel(text, mask, SENTINELS)
    target_text = nonnoise_span_to_unique_sentinel(text, mask, SENTINELS)
    return source_text, target_text


def span_corruption_mask(text, noise_density):

    text_length = len(text.split())
    num_masks = math.ceil(text_length * noise_density)

    mask = text_length * [0]
    padding = text_length * [0]
    possible_span_lengths = [text_length - i for i in range(text_length)]

    while num_masks > 0:

        max_possible_span_length = max(possible_span_lengths)

        if max_possible_span_length == 0:  # Merge spans
            possible_span_lengths = accumulate_possible_masks(
                padding
            )  # Assuming padding length == 1
            max_possible_span_length = max(possible_span_lengths)

        upper_span_length = min(min(MAX_SPAN_LENGTH, num_masks), max_possible_span_length)
        lower_span_length = min(min(MIN_SPAN_LENGTH, num_masks), max_possible_span_length)

        span_length = random.randint(lower_span_length, upper_span_length)

        possible_positions = [i for i, m in enumerate(possible_span_lengths) if m >= span_length]

        chosen_position = random.choice(possible_positions)

        for i in range(span_length):
            mask[chosen_position + i] = 1

        if chosen_position > 0:
            padding[chosen_position - 1] = 1

        if chosen_position + span_length < text_length:
            padding[chosen_position + span_length] = 1

        if chosen_position > 0:
            previous_possible_span_length = possible_span_lengths[chosen_position - 1]

        for i in range(span_length):
            possible_span_lengths[chosen_position + i] = 0
        if chosen_position > 0:
            possible_span_lengths[chosen_position - 1] = 0
        if chosen_position + span_length < text_length:
            possible_span_lengths[chosen_position + span_length] = 0

        j = 1
        while True:
            if chosen_position - 1 - j < 0:
                break
            if possible_span_lengths[chosen_position - 1 - j] == previous_possible_span_length + j:
                possible_span_lengths[chosen_position - 1 - j] = j
                j += 1
            else:
                break

        num_masks -= span_length

    return mask


def accumulate_possible_masks(possible_masks):
    possible_span_lengths = [0] * len(possible_masks)
    for i in range(len(possible_masks)):
        if possible_masks[i] == 0:
            continue
        accumulation = 1
        for j in range(1, len(possible_masks) - i):
            if possible_masks[i + j] == 1:
                accumulation += 1
            else:
                break
        possible_span_lengths[i] = accumulation
    return possible_span_lengths


def noise_span_to_unique_sentinel(text, mask, sentinels):
    tokens = text.split()
    text_ = []
    one_count = 0
    sentinel_cnt = 0
    for i in range(len(tokens)):
        if mask[i] == 1:
            one_count += 1
            if one_count == 1:
                text_.append(sentinels[sentinel_cnt])
                sentinel_cnt += 1
            else:
                if one_count == 3:
                    one_count = 0
        else:
            text_.append(tokens[i])
    text_ = " ".join(text_)
    return text_


def nonnoise_span_to_unique_sentinel(text, mask, sentinels):
    tokens = text.split()
    text_ = []
    zero_first = True
    sentinel_cnt = 0
    for i in range(len(tokens)):
        if mask[i] == 0:
            if zero_first:
                text_.append(sentinels[sentinel_cnt])
                zero_first = False
                sentinel_cnt += 1
        else:
            zero_first = True
            text_.append(tokens[i])
    text_ = " ".join(text_)
    return text_


# To replace converter as suggested in https://docs.python.org/3/library/logging.html#logging.Formatter.formatTime
def converter(self, timestamp):
    d = datetime.fromtimestamp(timestamp, tz=timezone("Asia/Seoul"))
    
    # From https://docs.python.org/ko/3/library/datetime.html#datetime.datetime.timetuple
    yday = d.toordinal() - date(d.year, 1, 1).toordinal() + 1
    dst = -1 if d.dst() is None else 1 if d.dst() != 0 else 0
    ct = time.struct_time((d.year, d.month, d.day,
                    d.hour, d.minute, d.second,
                    d.weekday(), yday, dst, d.tzname(), d.utcoffset().seconds))
    return ct
