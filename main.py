import typer
from typing import Optional
import os
import logging
import time
from datetime import datetime
from pytz import timezone
import logging

import torch
import numpy as np
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from datasets import load_dataset, load_metric
import wandb
from tqdm import tqdm
import json

from data import LanguageModelingDataset, StandardDataset, QuestionGenerationDataset, RandomMatrixDataset, RandomQuestionDataset
from utils import Method, Dataset, Optimizer
from utils import validate_arguments, parse_valid_file
from utils import DataCollatorForSeq2Seq
from utils import converter


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S %Z")
logging.Formatter.converter = converter


def main(
    method: Method = typer.Option(...),
    dataset: Dataset = typer.Option(..., help="Only SQuAD is supported for now."),
    chat: bool = typer.Option(False, help="Format for PersonaChat"),
    with_prompt: bool = typer.Option(False, help="Only for teacher evaluation. Concatenate prompt with input."),
    valid_path: str = typer.Option(
        ...,
        help="Validation dataset file. jsonl with id, prompt, input, and target fields. id can be '{prompt_id}-{input_id}' format.",
    ),
    model_path: str = typer.Option(
        ..., help="Model name or path for transformers' `from_pretrained`"
    ),
    teacher_path: Optional[str] = typer.Option(
        None,
        help="Only for distillation methods. Teacher model path for transformers' `from_pretrained`.",
    ),
    input_generator_path: Optional[str] = typer.Option(
        None,
        help="Only for distillation methods. Input generator model path for transformers' `from_pretrained`.",
    ),
    freeze_embeddings: bool = typer.Option(True, help="Freeze model token embeddings."),
    initial_noise: float = typer.Option(0.15, help="Only for LM-finetune. Initial density noise."),
    final_noise: float = typer.Option(0.15, help="Only for LM-finetune. Final density noise."),
    sample_temperature: float = typer.Option(1.0, help="Only for input generation method. Temerature of input generator."),
    questions_path: Optional[str] = typer.Option(
        None, help="Only for random question."
    ),
    input_max_length: int = typer.Option(
        30,
        help="Fixed input length. All inputs are padded & truncated to this length.",
    ),
    num_steps: Optional[int] = typer.Option(None, help="Number of steps to train. You can use either num_samples or num_steps."),
    num_samples: int = typer.Option(0, help="Number of samples to train on."),
    batch_size: int = typer.Option(8, help="Per-device batch size."),
    gradient_accumulation_steps: int = typer.Option(1),
    lr: float = typer.Option(1e-4, help="Learning rate."),
    optimizer_name: Optimizer = typer.Option(Optimizer.adamw),
    seed: int = typer.Option(42, help="Random seed."),
    log_every_steps: int = typer.Option(100, help="Log training progress every k steps"),
    valid_every_steps: int = typer.Option(100, help="Run validation every k steps"),
    print_valid_results: bool = typer.Option(False, help="Print every validation result"),
    save: bool = typer.Option(False, help="Save checkpoints"),
    port: int = typer.Option(12355, help="Port number for distributed data paralle."),
    output_dir: str = typer.Option("/outputs", help="Output directory"),
    name: str = typer.Option(..., help="Run name for wandb"),
):

    validate_arguments(method, dataset, valid_path)

    wandb.require("service")  # service improves wandb's handling of multiprocessing
    wandb.setup()

    world_size = torch.cuda.device_count()
    mp.spawn(
        train,
        nprocs=world_size,
        join=True,
        args=(
            world_size,
            method,
            dataset,
            chat,
            with_prompt,
            valid_path,
            model_path,
            teacher_path,
            input_generator_path,
            freeze_embeddings,
            initial_noise,
            final_noise,
            sample_temperature,
            questions_path,
            input_max_length,
            num_steps,
            num_samples,
            batch_size,
            gradient_accumulation_steps,
            lr,
            optimizer_name,
            seed,
            log_every_steps,
            valid_every_steps,
            print_valid_results,
            save,
            port,
            output_dir,
            name,
        ),
    )


def train(
    rank,
    world_size,
    method,
    dataset,
    chat,
    with_prompt,
    valid_path,
    model_path,
    teacher_path,
    input_generator_path,
    freeze_embeddings,
    initial_noise,
    final_noise,
    sample_temperature,
    questions_path,
    input_max_length,
    num_steps,
    num_samples,
    batch_size,
    gradient_accumulation_steps,
    lr,
    optimizer_name,
    seed,
    log_every_steps,
    valid_every_steps,
    print_valid_results,
    save,
    port,
    output_dir,
    name,
):
    setup(rank, world_size, port)

    set_seed(seed)

    if teacher_path is not None:
        teacher = AutoModelForSeq2SeqLM.from_pretrained(
            teacher_path,
        )
        teacher.to(rank)
    else:
        teacher = None

    if input_generator_path is not None:
        input_generator = AutoModelForSeq2SeqLM.from_pretrained(
            input_generator_path,
        )
        input_generator.to(rank)
    else:
        input_generator = None

    # Assume model, teacher, and input generator all use the same tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base",  # Temporary fix  # model_path
        use_fast=False,  # To prevent multiprocessing warning. cf) https://stackoverflow.com/a/67254879
    )

    effective_batch_size = batch_size * torch.cuda.device_count() * gradient_accumulation_steps
    if num_steps is None:
        num_steps = int(num_samples / effective_batch_size)
    if rank == 0:
        logging.info(f"Effective batch size (batch_size * num_devices * gradient_accumulation_steps): {effective_batch_size}")
        logging.info(f"Number of optimization steps: {num_steps}")

    valid_data = parse_valid_file(valid_path)
    metric = load_metric("squad")

    run_dir = f"{output_dir}/{name}"
    os.makedirs(run_dir, exist_ok=True)

    results = {}
    for pid, prompt_valid_data in tqdm(valid_data.items()):

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
        )
        model.to(rank)
        model = DDP(
            model, device_ids=[rank], find_unused_parameters=True if freeze_embeddings else False
        )

        if freeze_embeddings:
            model.module.encoder.embed_tokens.weight.requires_grad = False
            model.module.decoder.embed_tokens.weight.requires_grad = False
            model.module.lm_head.weight.requires_grad = False

        prompt = prompt_valid_data[0]["context"]
        valid_dataset = StandardDataset(prompt_valid_data, tokenizer, chat, with_prompt)
        # DataCollatorForSeq2Seq produces decoder_input_ids from labels
        valid_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            collate_fn=valid_data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            sampler=None,
        )

        # Default optimizer is AdamW for now. optimizer, betas, epsilon, and weight decay for groups can be adjusted.
        if optimizer_name == Optimizer.adam:
            optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer_name == Optimizer.adamw:
            optimizer = AdamW(model.parameters(), lr=lr)  # PyTorch implementation

        if rank == 0:
            wandb.init(
                project="persona",
                name=f"{name}/{pid}",
                dir=run_dir,
                config={
                    "method": method.value,
                    "dataset": dataset.value,
                    "chat": chat,
                    "with_prompt": with_prompt,
                    "valid-path": valid_path,
                    "model-path": model_path,
                    "teacher-path": teacher_path,
                    "input-generator-path": input_generator_path,
                    "freeze-embeddings": freeze_embeddings,
                    "sample_temperature": sample_temperature,
                    "input-max-length": input_max_length,
                    "num_steps": num_steps,
                    "num_samples": num_samples,
                    "batch_size": batch_size,
                    "gradient-accumulation-steps": gradient_accumulation_steps,
                    "lr": lr,
                    "optimizer_name": optimizer_name.value,
                    "seed": seed,
                },
                reinit=True,
            )

        if method == Method.lm_finetune:
            num_generations = batch_size * gradient_accumulation_steps * num_steps
            train_dataset = LanguageModelingDataset(rank, prompt, tokenizer, num_generations, initial_noise, final_noise)
            train_data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=train_data_collator,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            train_iterator = iter(train_dataloader)
        elif method == Method.input_generation:
            train_dataset = QuestionGenerationDataset(rank, prompt, tokenizer, chat)
            train_data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=train_data_collator,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            train_iterator = iter(train_dataloader)

            kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation
        elif method == Method.hybrid:
            num_generations = batch_size * gradient_accumulation_steps * num_steps
            train_data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            )
            lm_train_dataset = LanguageModelingDataset(rank, prompt, tokenizer, num_generations, initial_noise, final_noise)
            
            lm_train_dataloader = DataLoader(
                lm_train_dataset,
                batch_size=batch_size,
                collate_fn=train_data_collator,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            lm_train_iterator = iter(lm_train_dataloader)

            gen_train_dataset = QuestionGenerationDataset(rank, prompt, tokenizer, chat)
            gen_train_dataloader = DataLoader(
                gen_train_dataset,
                batch_size=batch_size,
                collate_fn=train_data_collator,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            gen_train_iterator = iter(gen_train_dataloader)

            kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation

        elif method == Method.random_matrix:
            model.module.encoder.embed_tokens.weight.requires_grad = False
            model.module.decoder.embed_tokens.weight.requires_grad = False
            model.module.lm_head.weight.requires_grad = False
            
            m = torch.mean(model.module.shared.weight, dim=0)
            s = torch.std(model.module.shared.weight, dim=0)
            train_dataset = RandomMatrixDataset(prompt, tokenizer, 30, 512, torch.mean(model.module.shared.weight, dim=0), torch.std(model.module.shared.weight, dim=0))
            train_data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=train_data_collator,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            train_iterator = iter(train_dataloader)

            kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation

        elif method == Method.random_question:
            questions = []
            with open(questions_path) as f:
                for line in f:
                    j = json.loads(line)
                    questions.append(j["input"])
            train_dataset = RandomQuestionDataset(questions, prompt, tokenizer, chat)
            train_data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8,
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                collate_fn=train_data_collator,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            train_iterator = iter(train_dataloader)

            kl_criterion = nn.KLDivLoss(reduction="batchmean")  # For loss calculation
            
        iteration_step = 0
        optimization_step = 0

        while True:
            if method == Method.evaluate_only:
                break

            model.train()

            if method == Method.lm_finetune:
                batch = next(train_iterator)
                batch = {k: v.to(rank) for k, v in batch.items()}

                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = outputs.loss
            if method == Method.input_generation:
                batch = next(train_iterator)
                batch = {k: v.to(rank) for k, v in batch.items()}
                
                with torch.no_grad():
                    generation_outputs = input_generator.generate(
                        input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"],
                        do_sample=True,  # This causes difference between sequences and argmax of outputs
                        temperature=sample_temperature,
                        return_dict_in_generate=True,
                        max_length=input_max_length,
                    )
                    generated_input = generation_outputs["sequences"][:, 1:]  # Exclude SOS

                    teacher_input_ids = torch.cat([batch["input_prompt_ids"], generated_input, batch["input_ids"]], dim=1)
                    teacher_mask = teacher_input_ids > 1  # Exclude padding (0) and EOS (1)

                    teacher_outputs = teacher.generate(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_length=input_max_length,
                    )

                    teacher_scores = []
                    for position in teacher_outputs["scores"]:
                        teacher_scores.append(position)
                    teacher_scores = torch.stack(teacher_scores, dim=1)

                student_input_ids = torch.cat([batch["input_prompt_ids"], generated_input], dim=1)
                student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

                student_outputs = model(
                    input_ids=student_input_ids,
                    decoder_input_ids=teacher_outputs["sequences"][:, :-1],
                    attention_mask=student_mask,
                    output_hidden_states=True)

                labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
                labels[labels == tokenizer.pad_token_id] = -100
                logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)

                student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
                teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask

                temperature = 1
                loss_ce = (
                    kl_criterion(
                        nn.functional.log_softmax(student_logits_selected / temperature, dim=-1),
                        nn.functional.softmax(teacher_logits_selected / temperature, dim=-1),
                    )
                    * (temperature) ** 2
                )
                loss = loss_ce
            if method == Method.hybrid:
                lm_batch = next(lm_train_iterator)
                lm_batch = {k: v.to(rank) for k, v in lm_batch.items()}
                
                lm_outputs = model(input_ids=lm_batch["input_ids"], attention_mask=lm_batch["attention_mask"], labels=lm_batch["labels"])
                lm_loss = lm_outputs.loss

                gen_batch = next(gen_train_iterator)
                gen_batch = {k: v.to(rank) for k, v in gen_batch.items()}
                
                with torch.no_grad():
                    generation_outputs = input_generator.generate(
                        input_ids=gen_batch["input_ids"], 
                        attention_mask=gen_batch["attention_mask"],
                        do_sample=True,  # This causes difference between sequences and argmax of outputs
                        temperature=sample_temperature,
                        return_dict_in_generate=True,
                        max_length=input_max_length,
                    )
                    generated_input = generation_outputs["sequences"][:, 1:]  # Exclude SOS

                    teacher_input_ids = torch.cat([gen_batch["input_prompt_ids"], generated_input, gen_batch["input_ids"]], dim=1)
                    teacher_mask = teacher_input_ids > 1  # Exclude padding (0) and EOS (1)

                    teacher_outputs = teacher.generate(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_length=input_max_length,
                    )

                    teacher_scores = []
                    for position in teacher_outputs["scores"]:
                        teacher_scores.append(position)
                    teacher_scores = torch.stack(teacher_scores, dim=1)

                student_input_ids = torch.cat([gen_batch["input_prompt_ids"], generated_input], dim=1)
                student_mask = student_input_ids > 1  # Exclude padding (0) and EOS (1)

                student_outputs = model(
                    input_ids=student_input_ids,
                    decoder_input_ids=teacher_outputs["sequences"][:, :-1],
                    attention_mask=student_mask,
                    output_hidden_states=True)

                labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
                labels[labels == tokenizer.pad_token_id] = -100
                logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)

                student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
                teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask

                temperature = 1
                loss_ce = (
                    kl_criterion(
                        nn.functional.log_softmax(student_logits_selected / temperature, dim=-1),
                        nn.functional.softmax(teacher_logits_selected / temperature, dim=-1),
                    )
                    * (temperature) ** 2
                )
                
                loss = (lm_loss + loss_ce) / 2
                
            elif method == Method.random_matrix:

                batch = next(train_iterator)
                batch = {k: v.to(rank) for k, v in batch.items()}
                
                random_matrix = torch.stack([torch.stack([torch.normal(m, s) for _ in range(30)]) for _ in range(batch["input_prompt_ids"].size(0))]).to('cuda')
                input_prompt_embeds = model.module.get_input_embeddings()(batch["input_prompt_ids"])
                    
                with torch.no_grad():
                    context_embeds = model.module.get_input_embeddings()(batch["input_ids"])
                
                    input_context_embeds = torch.cat([input_prompt_embeds, random_matrix, context_embeds], dim=1)
                    input_context_attention_mask = torch.cat([batch["input_attention_mask"], batch["attention_mask"]], dim=1)
                    assert input_context_embeds.size(1) == input_context_attention_mask.size(-1)
                    
                    teacher_outputs = teacher.generate(
                        inputs_embeds=input_context_embeds,
                        attention_mask=input_context_attention_mask,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_length=input_max_length,
                    )

                    teacher_scores = []
                    for position in teacher_outputs["scores"]:
                        teacher_scores.append(position)
                    teacher_scores = torch.stack(teacher_scores, dim=1)

                input_embeds = torch.cat([input_prompt_embeds, random_matrix], dim=1)
                
                student_outputs = model(
                    inputs_embeds=input_embeds,
                    attention_mask=batch["input_attention_mask"],
                    decoder_input_ids=teacher_outputs["sequences"][:, :-1],
                    output_hidden_states=True)

                labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
                labels[labels == tokenizer.pad_token_id] = -100
                logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)

                student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
                teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask

                temperature = 1
                loss_ce = (
                    kl_criterion(
                        nn.functional.log_softmax(student_logits_selected / temperature, dim=-1),
                        nn.functional.softmax(teacher_logits_selected / temperature, dim=-1),
                    )
                    * (temperature) ** 2
                )
                loss = loss_ce
            
            if method == Method.random_question:
                batch = next(train_iterator)
                batch = {k: v.to(rank) for k, v in batch.items()}
                
                with torch.no_grad():
                    teacher_outputs = teacher.generate(
                        input_ids=batch["teacher_input_ids"],
                        attention_mask=batch["teacher_attention_mask"],
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_length=input_max_length,
                    )

                    teacher_scores = []
                    for position in teacher_outputs["scores"]:
                        teacher_scores.append(position)
                    teacher_scores = torch.stack(teacher_scores, dim=1)

                student_outputs = model(
                    input_ids=batch["input_ids"],
                    decoder_input_ids=teacher_outputs["sequences"][:, :-1],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True)

                labels = teacher_outputs["sequences"][:, 1:]  # Exclude SOS
                labels[labels == tokenizer.pad_token_id] = -100
                logits_mask = (labels > -1).unsqueeze(-1).expand_as(student_outputs.logits)

                student_logits_selected = torch.masked_select(student_outputs.logits, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                student_logits_selected = student_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
                teacher_logits_selected = torch.masked_select(teacher_scores, logits_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                teacher_logits_selected = teacher_logits_selected.view(-1, student_outputs.logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask

                temperature = 1
                loss_ce = (
                    kl_criterion(
                        nn.functional.log_softmax(student_logits_selected / temperature, dim=-1),
                        nn.functional.softmax(teacher_logits_selected / temperature, dim=-1),
                    )
                    * (temperature) ** 2
                )
                loss = loss_ce

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            iteration_step += 1

            if iteration_step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                optimization_step += 1

                if rank == 0:
                    wandb.log(
                        {
                            "train/loss": loss.item() * gradient_accumulation_steps,
                        },
                        step=optimization_step,
                    )

            if optimization_step == num_steps:
                break

            if iteration_step % (gradient_accumulation_steps * valid_every_steps) == 0:
                if rank == 0:
                    metrics = run_validation(
                        model, rank, prompt_valid_data, valid_dataloader, tokenizer, metric
                    )
                    if print_valid_results:
                        logging.info(f"{name}/{pid}/{optimization_step}: {metrics}")
                    wandb.log(
                        {
                            "valid/exact_match": metrics["exact_match"],
                            "valid/f1": metrics["f1"],
                            "valid/perplexity": metrics["perplexity"],
                        },
                        step=optimization_step,
                    )
                dist.barrier()

        if rank == 0:
            metrics = run_validation(
                model, rank, prompt_valid_data, valid_dataloader, tokenizer, metric
            )
            results[pid] = metrics

            logging.info(f"{name}/{pid}: {metrics}")
            wandb.log(
                {
                    "valid/exact_match": metrics["exact_match"],
                    "valid/f1": metrics["f1"],
                    "valid/perplexity": metrics["perplexity"],
                },
                step=optimization_step,
            )
            wandb.summary["exact_match"] = metrics["exact_match"]
            wandb.summary["f1"] = metrics["f1"]
            wandb.summary["perplexity"] = metrics["perplexity"]
            wandb.finish()

            if save:
                checkpoint_dir = f"{output_dir}/{name}/checkpoints/{pid}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.module.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

        dist.barrier()

    if rank == 0:
        em_mean = sum([metric["exact_match"] for metric in results.values()]) / len(results)
        f1_mean = sum([metric["f1"] for metric in results.values()]) / len(results)
        perplexity_mean = sum([metric["perplexity"] for metric in results.values()]) / len(results)

        print()
        print(f"em mean {em_mean:0.4f}")
        print(f"f1 mean {f1_mean:0.4f}")
        print(f"perplexity {perplexity_mean:0.4f}")

        wandb.alert(
            title=f"Run finished",
            text=f"Run {name} just finished. EM: {em_mean}, F1: {f1_mean}",
        )
    
    dist.barrier()
    cleanup()


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = f"{port}"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_validation(model, rank, prompt_valid_data, valid_dataloader, tokenizer, metric):
    model.eval()
    all_predictions = []
    all_losses = []
    with torch.no_grad():
        for batch in valid_dataloader:
            batch = {k: v.to(rank) for k, v in batch.items()}

            labels = batch.pop("labels")
            generations = model.module.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            predictions = tokenizer.batch_decode(
                generations, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            all_predictions.extend(predictions)

            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            all_losses.append(loss.item())

    rank_prompt_valid_data = []
    for sample_index in valid_dataloader.sampler:
        rank_prompt_valid_data.append(prompt_valid_data[sample_index])

    references = [
        {"id": datum["id"], "answers": {"text": [datum["target"]], "answer_start": [0]}}
        for datum in rank_prompt_valid_data
    ]
    predictions = [
        {"id": rank_prompt_valid_data[i]["id"], "prediction_text": prediction}
        for i, prediction in enumerate(all_predictions)
    ]

    metrics = metric.compute(predictions=predictions, references=references)

    mean_loss = sum(all_losses)/len(all_losses)
    perplexity = np.exp(mean_loss)
    metrics["perplexity"] = perplexity

    return metrics


if __name__ == "__main__":
    typer.run(main)
