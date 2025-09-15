"""
squad_newsqa_pipeline.py
--------------------------------------

This script demonstrates how to perform domain adaptation and knowledge
distillation for machine reading comprehension using the SQuAD and
NewsQA datasets.  The approach mirrors the MEC‑DA methodology applied
to question answering.  A large teacher model (BERT) is first trained
on the source domain (SQuAD) while simultaneously encouraging
domain‑invariant representations through an adversarial domain
classifier.  The adapted teacher is then used to generate soft labels
on both the source and target domains (NewsQA) to train a compact
student model (e.g. DistilBERT).

Background
==========

* **SQuAD (Stanford Question Answering Dataset)** is a reading
  comprehension dataset consisting of questions posed by crowdworkers
  on a set of Wikipedia articles【779973264818019†L12-L16】.  Each answer is a
  contiguous span of text in the corresponding passage.  SQuAD 1.1
  contains over 100,000 question‑answer pairs【779973264818019†L30-L31】.

* **NewsQA** is a machine comprehension dataset of over 100,000
  human‑generated question‑answer pairs collected from more than 10,000
  CNN news articles【410989956311321†L49-L55】.  Answers consist of text spans,
  and the dataset was designed to elicit questions requiring
  reasoning【410989956311321†L49-L55】.

The goal of domain adaptation is to train a model on the labeled
SQuAD source data so that it performs well on the NewsQA target data
without using any NewsQA labels.  We achieve this by:

1. **Domain‑adversarial training**: augment a BERT‑based QA model with
   a gradient reversal layer and a domain classifier.  The model
   learns to answer SQuAD questions while being unable to distinguish
   between SQuAD and NewsQA features, thus making the internal
   representations domain invariant.
2. **Knowledge distillation**: use the adapted teacher to generate
   soft start/end distributions for both domains and train a compact
   student model via Kullback–Leibler divergence combined with the
   supervised SQuAD loss.

The code below provides a skeleton implementation that illustrates
these stages.  It relies on PyTorch and the Hugging Face Transformers
and Datasets libraries.  Because this environment does not execute
Python, the script is meant as a guideline and has not been tested
here.
"""

import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoConfig,
    get_linear_schedule_with_warmup,
    default_data_collator,
)
from datasets import load_dataset


###############################################################################
# Data preprocessing
###############################################################################

def prepare_train_features(examples, tokenizer, max_length: int = 384, doc_stride: int = 128):
    """Convert raw SQuAD/NewsQA examples into model inputs.

    This function is adapted from the Hugging Face QA training scripts.  It
    tokenizes the question and context, computes offset mappings and locates
    the start and end token indices for answer spans.  For target domain
    examples (NewsQA) without labels, the answer indices remain -1.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # Each example may map to multiple features if it is long
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    start_positions = []
    end_positions = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If there is no answer, set start/end positions to CLS
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            # Find the start and end token indices within the sequence
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # If the answer is outside the span, set to CLS
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Otherwise find the token indices within offsets
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def load_squad_newsqa(tokenizer, max_length: int = 384, doc_stride: int = 128) -> Tuple:
    """Load SQuAD and NewsQA datasets and prepare features.

    Returns:
        squad_train, squad_val, newsqa_train, newsqa_val datasets in
        PyTorch‑friendly format.
    """
    # Load datasets
    squad = load_dataset("squad")
    newsqa = load_dataset("newsqa", split="train").train_test_split(test_size=0.1)
    squad_train_raw = squad["train"]
    squad_val_raw = squad["validation"]
    newsqa_train_raw = newsqa["train"]
    newsqa_val_raw = newsqa["test"]
    # Prepare features
    squad_train = squad_train_raw.map(
        lambda x: prepare_train_features(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=squad_train_raw.column_names,
    )
    squad_val = squad_val_raw.map(
        lambda x: prepare_train_features(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=squad_val_raw.column_names,
    )
    # For NewsQA, answers may be empty if unlabeled; treat the same
    newsqa_train = newsqa_train_raw.map(
        lambda x: prepare_train_features(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=newsqa_train_raw.column_names,
    )
    newsqa_val = newsqa_val_raw.map(
        lambda x: prepare_train_features(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=newsqa_val_raw.column_names,
    )
    return squad_train, squad_val, newsqa_train, newsqa_val


###############################################################################
# Domain‑adversarial QA model
###############################################################################

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, lambda_):
        ctx.lambda_ = lambda_
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, input_):
        return GradientReversalFunction.apply(input_, self.lambda_)


class QADANNModel(nn.Module):
    """BERT‑based QA model with a domain classifier and gradient reversal."""

    def __init__(self, encoder_name: str = "bert-base-uncased", lambda_da: float = 0.1):
        super().__init__()
        self.lambda_da = lambda_da
        self.encoder_config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name, config=self.encoder_config)
        hidden_size = self.encoder_config.hidden_size
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.grl = GradientReversalLayer(lambda_da)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        reversed_features = self.grl(pooled_output)
        domain_logits = self.domain_classifier(reversed_features)
        return start_logits, end_logits, domain_logits


###############################################################################
# Training and evaluation
###############################################################################

def train_qadann(
    model: QADANNModel,
    squad_loader: DataLoader,
    newsqa_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int = 2,
    lambda_da: float = 0.1,
) -> None:
    """Train the domain‑adversarial QA model on SQuAD and NewsQA.

    Only SQuAD examples provide supervised QA loss (start and end positions).
    NewsQA examples contribute to the domain classifier but are treated as
    unlabeled for QA.  The domain labels are 0 for SQuAD and 1 for NewsQA.
    """
    model.to(device)
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        # Iterators for both datasets
        squad_iter = iter(squad_loader)
        newsqa_iter = iter(newsqa_loader)
        num_batches = max(len(squad_loader), len(newsqa_loader))
        for _ in range(num_batches):
            try:
                s_batch = next(squad_iter)
            except StopIteration:
                squad_iter = iter(squad_loader)
                s_batch = next(squad_iter)
            try:
                n_batch = next(newsqa_iter)
            except StopIteration:
                newsqa_iter = iter(newsqa_loader)
                n_batch = next(newsqa_iter)
            # Concatenate inputs from both domains
            input_ids = torch.cat([s_batch["input_ids"], n_batch["input_ids"]], dim=0).to(device)
            attention_mask = torch.cat([s_batch["attention_mask"], n_batch["attention_mask"]], dim=0).to(device)
            # Forward pass
            start_logits, end_logits, domain_logits = model(input_ids, attention_mask)
            # QA loss on SQuAD portion only
            s_len = s_batch["input_ids"].size(0)
            qa_start = start_logits[:s_len]
            qa_end = end_logits[:s_len]
            loss_start = ce_loss(qa_start, s_batch["start_positions"].to(device))
            loss_end = ce_loss(qa_end, s_batch["end_positions"].to(device))
            loss_qa = (loss_start + loss_end) / 2
            # Domain loss on all examples
            domain_labels = torch.cat([
                torch.zeros(s_len, dtype=torch.long),
                torch.ones(n_batch["input_ids"].size(0), dtype=torch.long),
            ], dim=0).to(device)
            loss_domain = ce_loss(domain_logits, domain_labels)
            loss = loss_qa + lambda_da * loss_domain
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} – Loss: {avg:.4f}")


def predict_start_end(model, dataloader: DataLoader, device: torch.device) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute teacher start/end probability distributions for distillation."""
    model.eval()
    all_starts = []
    all_ends = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_logits, end_logits, _ = model(input_ids, attention_mask)
            start_probs = torch.nn.functional.softmax(start_logits, dim=-1)
            end_probs = torch.nn.functional.softmax(end_logits, dim=-1)
            all_starts.append(start_probs.cpu())
            all_ends.append(end_probs.cpu())
    return all_starts, all_ends


def train_student_qa(
    teacher: QADANNModel,
    student: AutoModelForQuestionAnswering,
    squad_loader: DataLoader,
    newsqa_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int = 2,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> None:
    """Train a student QA model via knowledge distillation.

    Args:
        teacher: Trained adversarial teacher model.
        student: Compact QA model with QA head.
        squad_loader: Labeled SQuAD dataloader for supervised loss.
        newsqa_loader: Unlabeled NewsQA dataloader for distillation.
        alpha: Weight for distillation loss versus supervised loss.
        temperature: Temperature for distillation.
    """
    teacher.to(device)
    teacher.eval()
    student.to(device)
    student.train()
    ce_loss = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        squad_iter = iter(squad_loader)
        newsqa_iter = iter(newsqa_loader)
        num_batches = max(len(squad_loader), len(newsqa_loader))
        for _ in range(num_batches):
            try:
                s_batch = next(squad_iter)
            except StopIteration:
                squad_iter = iter(squad_loader)
                s_batch = next(squad_iter)
            try:
                n_batch = next(newsqa_iter)
            except StopIteration:
                newsqa_iter = iter(newsqa_loader)
                n_batch = next(newsqa_iter)
            # Concatenate source and target inputs
            input_ids = torch.cat([s_batch["input_ids"], n_batch["input_ids"]], dim=0).to(device)
            attention_mask = torch.cat([s_batch["attention_mask"], n_batch["attention_mask"]], dim=0).to(device)
            # Teacher predictions
            with torch.no_grad():
                t_start_logits, t_end_logits, _ = teacher(input_ids, attention_mask)
                t_start_probs = torch.nn.functional.softmax(t_start_logits / temperature, dim=-1)
                t_end_probs = torch.nn.functional.softmax(t_end_logits / temperature, dim=-1)
            # Student predictions
            outputs = student(input_ids=input_ids, attention_mask=attention_mask)
            s_start_logits, s_end_logits = outputs.start_logits, outputs.end_logits
            s_start_log_probs = torch.nn.functional.log_softmax(s_start_logits / temperature, dim=-1)
            s_end_log_probs = torch.nn.functional.log_softmax(s_end_logits / temperature, dim=-1)
            # Distillation KL divergence on both domains
            loss_kd_start = torch.nn.functional.kl_div(s_start_log_probs, t_start_probs, reduction="batchmean") * (temperature ** 2)
            loss_kd_end = torch.nn.functional.kl_div(s_end_log_probs, t_end_probs, reduction="batchmean") * (temperature ** 2)
            loss_kd = (loss_kd_start + loss_kd_end) / 2
            # Supervised loss on source portion only
            s_len = s_batch["input_ids"].size(0)
            sup_start_logits = s_start_logits[:s_len]
            sup_end_logits = s_end_logits[:s_len]
            sup_loss_start = ce_loss(sup_start_logits, s_batch["start_positions"].to(device))
            sup_loss_end = ce_loss(sup_end_logits, s_batch["end_positions"].to(device))
            loss_sup = (sup_loss_start + sup_loss_end) / 2
            loss = alpha * loss_kd + (1.0 - alpha) * loss_sup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg = total_loss / num_batches
        print(f"Student Epoch {epoch+1}/{epochs} – Loss: {avg:.4f}")


###############################################################################
# Main function
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Domain adaptation and KD for QA")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs_da", type=int, default=2, help="Epochs for adversarial training")
    parser.add_argument("--epochs_kd", type=int, default=2, help="Epochs for knowledge distillation")
    parser.add_argument("--lambda_da", type=float, default=0.1, help="Weight for domain adversarial loss")
    parser.add_argument("--alpha_kd", type=float, default=0.5, help="Weight for distillation loss")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--teacher_model", type=str, default="bert-base-uncased")
    parser.add_argument("--student_model", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()
    device = torch.device(args.device)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    # Load and prepare datasets
    squad_train, squad_val, newsqa_train, newsqa_val = load_squad_newsqa(tokenizer, args.max_length, args.doc_stride)
    # Create dataloaders
    squad_train_loader = DataLoader(squad_train, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)
    squad_val_loader = DataLoader(squad_val, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)
    newsqa_train_loader = DataLoader(newsqa_train, batch_size=args.batch_size, shuffle=True, collate_fn=default_data_collator)
    newsqa_val_loader = DataLoader(newsqa_val, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)
    # Stage 1: Train teacher
    print("=== Stage 1: Adversarial QA Training ===")
    teacher = QADANNModel(encoder_name=args.teacher_model, lambda_da=args.lambda_da)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=3e-5)
    total_steps = args.epochs_da * max(len(squad_train_loader), len(newsqa_train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_qadann(teacher, squad_train_loader, newsqa_train_loader, optimizer, scheduler, device, epochs=args.epochs_da, lambda_da=args.lambda_da)
    # Stage 2: Knowledge distillation
    print("=== Stage 2: Knowledge Distillation ===")
    student = AutoModelForQuestionAnswering.from_pretrained(args.student_model)
    optimizer_s = torch.optim.AdamW(student.parameters(), lr=3e-5)
    total_steps_kd = args.epochs_kd * max(len(squad_train_loader), len(newsqa_train_loader))
    scheduler_s = get_linear_schedule_with_warmup(optimizer_s, num_warmup_steps=0, num_training_steps=total_steps_kd)
    train_student_qa(teacher, student, squad_train_loader, newsqa_train_loader, optimizer_s, scheduler_s, device, epochs=args.epochs_kd, alpha=args.alpha_kd, temperature=args.temperature)
    # Optionally evaluate on NewsQA (requires computing F1/EM metrics) – omitted for brevity


if __name__ == "__main__":
    main()