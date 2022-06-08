import random

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

from utils import generate_masked_sample


class RandomMatrixDataset(IterableDataset):
    def __init__(self, context, tokenizer, input_length, hidden_size, mean, std):
        input_prompt = "question: "
        self.input_prompt_inputs = tokenizer(input_prompt)
        self.input_length = input_length

        context_sequence = f"context: {context}"
        self.context_inputs = tokenizer(
            context_sequence,
            truncation=True,
        )
        self.hidden_size = hidden_size
        self.mean = mean
        self.std = std

    def __iter__(self):
        while True:
            model_inputs = {}

            # random_matrix = torch.stack([torch.normal(self.mean, self.std) for _ in range(self.input_length)])
            # model_inputs["random_matrix"] = random_matrix
            model_inputs["input_prompt_ids"] = self.input_prompt_inputs["input_ids"][:-1]  # Exclude EOS
            model_inputs["input_attention_mask"] = [1] * (len(model_inputs["input_prompt_ids"]) + self.input_length)
            model_inputs["input_ids"] = self.context_inputs["input_ids"]
            model_inputs["attention_mask"] = self.context_inputs["attention_mask"]
            yield model_inputs


class RandomQuestionDataset(IterableDataset):
    def __init__(self, questions, context, tokenizer, chat):
        def prepare_input(question):
            model_input = {}

            if not chat:
                question_sequence = f"question: {question.lstrip()}"
            else:
                question_sequence = f"history: {question.lstrip()}"
            question_input = tokenizer(
                question_sequence,
                truncation=True,
            )
            model_input["input_ids"] = question_input["input_ids"]
            model_input["attention_mask"] = question_input["attention_mask"]

            if not chat:
                teacher_sequence = f"question: {question.lstrip()} context: {context}"
            else:
                teacher_sequence = f"history: {question.lstrip()} persona: {context}"
            teacher_input = tokenizer(
                teacher_sequence,
                truncation=True,
            )
            model_input["teacher_input_ids"] = teacher_input["input_ids"]
            model_input["teacher_attention_mask"] = teacher_input["attention_mask"]
            return model_input

        self.inputs = [prepare_input(question) for question in questions]

    def __iter__(self):
        while True:
            model_inputs = random.choice(self.inputs)
            yield model_inputs


class LanguageModelingDataset(IterableDataset):
    def __init__(self, rank, context, tokenizer, num_generations, initial_noise, final_noise):
        self.rank = rank
        self.context = context
        self.tokenizer = tokenizer
        self.num_generations = num_generations
        self.initial_noise = initial_noise
        self.final_noise = final_noise

    def __iter__(self):
        sample_index = 0
        while True:
            current_noise = self.initial_noise + (self.final_noise - self.initial_noise) * sample_index / self.num_generations
            source_text, target_text = generate_masked_sample(self.context, current_noise)
            
            source_inputs = self.tokenizer(source_text, truncation=True)
            target_inputs = self.tokenizer(target_text, truncation=True)
            
            model_inputs = {
                "input_ids": source_inputs["input_ids"],
                "attention_mask": source_inputs["attention_mask"],
                "labels": target_inputs["input_ids"],
                "labels_attention_mask": target_inputs["attention_mask"],
            }

            yield model_inputs
            sample_index += 1


class StandardDataset(Dataset):
    def __init__(self, data, tokenizer, chat, with_prompt):
        self.data = data
        self.tokenizer = tokenizer
        self.chat = chat
        self.with_prompt = with_prompt

    def __getitem__(self, item):
        datum = self.data[item]
        model_inputs = {}
        if self.chat:
            key = "history:"
            prompt = " ".join(["persona:", datum["context"].lstrip()])
        else:
            key = "question:"
            prompt = " ".join(["context:", datum["context"].lstrip()])
        if self.with_prompt:
            input_text = " ".join([key, datum["input"].lstrip(), prompt])
        else:
            input_text = " ".join([key, datum["input"].lstrip()])
        inputs = self.tokenizer(input_text)
        model_inputs["input_ids"] = inputs["input_ids"]
        model_inputs["attention_mask"] = inputs["attention_mask"]
        model_inputs["labels"] = self.tokenizer(datum["target"])["input_ids"]
        return model_inputs

    def __len__(self):
        return len(self.data)


class QuestionGenerationDataset(IterableDataset):
    def __init__(self, rank, context, tokenizer, chat):
        self.model_inputs = {}

        if not chat:
            input_prompt = "question: "
            context_sequence = f"context: {context}"
        else:
            input_prompt = "history: "
            context_sequence = f"persona: {context}"
        context_inputs = tokenizer(context_sequence, truncation=True)
        self.model_inputs["input_ids"] = context_inputs["input_ids"]
        self.model_inputs["attention_mask"] = context_inputs["attention_mask"]
        
        input_prompt_tokenized = tokenizer(input_prompt)

        self.model_inputs["input_prompt_ids"] = input_prompt_tokenized["input_ids"][:-1]  # Exclude EOS
        
    def __iter__(self):
        while True:
            yield self.model_inputs
