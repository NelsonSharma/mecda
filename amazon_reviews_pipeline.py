"""
amazon_reviews_pipeline.py
-------------------------------------------

This script provides an end‑to‑end pipeline for performing domain adaptation
and knowledge distillation on the classic Amazon Reviews sentiment dataset.
The goal is to mimic the high‑level stages of the MEC‑DA framework—
adaptation of a large model on a source domain followed by distillation
into a smaller model—while operating on text data rather than images.

The code is structured to work with the Multi‑Domain Sentiment Dataset
released by Blitzer et al., which contains labeled and unlabeled reviews
from four product categories (Books, DVDs, Electronics and Kitchen)
【630686179138186†L233-L244】.  Each domain includes roughly 2 000 labeled reviews
and around 4 000 unlabeled ones.  Ratings above three are treated as
positive and ratings below three as negative, yielding a binary
sentiment classification problem【630686179138186†L239-L244】.

Key components of this script:

  • Data download and extraction: the dataset is fetched from the
    Johns Hopkins University website, extracted and parsed into
    training and evaluation splits.
  • Domain‑adversarial training (DANN): a neural network with a
    gradient reversal layer learns domain‑invariant features.  A
    pre‑trained BERT encoder produces representations that feed into
    both a sentiment classifier and a domain classifier.  During
    training the sentiment loss and domain loss are optimized jointly
    to encourage the encoder to learn representations that cannot
    easily distinguish between the source and target domains.
  • Knowledge distillation: after training the large model (e.g., BERT)
    with domain adaptation, its softened outputs on both source and
    target data are used to train a smaller student model (e.g.,
    DistilBERT).  The student learns from the teacher’s soft labels
    using Kullback–Leibler divergence while also being supervised on
    the available labeled source examples.  This mirrors the
    collaborative knowledge distillation stage of MEC‑DA in the
    language domain.

The script is parameterized by command‑line arguments so that
different domain pairs, models and hyper‑parameters can be specified
without editing the code.  See the `if __name__ == "__main__":`
section for usage details.

NOTE:  This implementation is provided as a reference and has not
been executed in this environment.  It relies on PyTorch and the
Hugging Face Transformers library.  Training large language models
requires a GPU and may take several hours.  Adjust batch sizes and
learning rates based on your available compute resources.
"""

import argparse
import os
import re
import tarfile
import urllib.request
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


###############################################################################
# Data loading and preprocessing
###############################################################################

def download_mdsd(destination_dir: str) -> str:
    """Download the Multi‑Domain Sentiment Dataset and return the archive path.

    The dataset contains reviews from four domains—books, DVDs, electronics and
    kitchen appliances—with both positive and negative examples.  Only a
    subset of the full Amazon corpus is used here【630686179138186†L233-L244】.

    Args:
        destination_dir: Directory where the archive will be saved.

    Returns:
        Path to the downloaded tar.gz archive.
    """
    os.makedirs(destination_dir, exist_ok=True)
    url = "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz"
    archive_path = os.path.join(destination_dir, "domain_sentiment_data.tar.gz")
    if not os.path.exists(archive_path):
        print(f"Downloading dataset from {url} …")
        urllib.request.urlretrieve(url, archive_path)
        print(f"Saved archive to {archive_path}")
    else:
        print(f"Archive already exists at {archive_path}")
    return archive_path


def extract_mdsd(archive_path: str, extract_to: str) -> None:
    """Extract the Multi‑Domain Sentiment Dataset archive to a given directory."""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted archive to {extract_to}")


def parse_review_block(review_block: str) -> Tuple[str, int]:
    """Extract the review text and sentiment label from a pseudo‑XML review.

    The dataset uses a simple XML format where each review is wrapped in
    `<review>` tags and contains fields like `<rating>`, `<review_text>`, etc.
    A rating greater than 3 is considered positive (1), while a rating less
    than 3 is negative (0)【630686179138186†L239-L244】.  Ratings equal to 3 are
    ignored.

    Args:
        review_block: A string containing the review XML.

    Returns:
        A tuple of (review text, sentiment label).
    """
    # Extract rating
    rating_match = re.search(r"<rating>(\d)</rating>", review_block)
    if not rating_match:
        return None  # skip malformed review
    rating = int(rating_match.group(1))
    if rating == 3:
        return None
    label = 1 if rating > 3 else 0
    # Extract review text; fall back to <review_text> if present, otherwise empty
    text_match = re.search(r"<review_text>(.*?)</review_text>", review_block, re.DOTALL)
    review_text = text_match.group(1).strip() if text_match else ""
    return review_text, label


def load_domain_dataset(domain_dir: str, max_labeled: int = 2000) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load labeled and unlabeled reviews from a specific domain.

    Each domain directory contains three files: positive.review,
    negative.review and unlabeled.review【630686179138186†L233-L244】.  This function
    reads the positive and negative files up to a maximum number of labeled
    examples and also returns unlabeled reviews for unsupervised domain
    adaptation.

    Args:
        domain_dir: Path to the domain directory.
        max_labeled: Maximum number of labeled examples to load per domain.

    Returns:
        A tuple (labeled_texts, labels, unlabeled_texts, unlabeled_labels).  The
        unlabeled_labels list contains dummy labels (e.g., -1) because they are
        unknown.
    """
    labeled_texts: List[str] = []
    labels: List[int] = []
    # Load positive and negative reviews
    for label_name, filename in [("pos", "positive.review"), ("neg", "negative.review")]:
        file_path = os.path.join(domain_dir, filename)
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            content = f.read()
        # Split file into individual reviews; reviews are separated by newlines and
        # each review starts with "<review>"
        review_blocks = re.findall(r"<review>(.*?)</review>", content, re.DOTALL)
        for block in review_blocks:
            parsed = parse_review_block(block)
            if parsed is None:
                continue
            text, label = parsed
            labeled_texts.append(text)
            labels.append(label)
            if len(labeled_texts) >= max_labeled:
                break
        if len(labeled_texts) >= max_labeled:
            break
    # Load unlabeled reviews (useful for unsupervised adaptation); labels set to -1
    unlabeled_texts: List[str] = []
    unlabeled_labels: List[int] = []
    unlabeled_file = os.path.join(domain_dir, "unlabeled.review")
    if os.path.exists(unlabeled_file):
        with open(unlabeled_file, "r", encoding="ISO-8859-1") as f:
            content = f.read()
        review_blocks = re.findall(r"<review>(.*?)</review>", content, re.DOTALL)
        for block in review_blocks:
            text_label = parse_review_block(block)
            # Unlabeled reviews may not contain ratings; if so, treat as unlabeled
            if text_label is None:
                # Extract review text
                text_match = re.search(r"<review_text>(.*?)</review_text>", block, re.DOTALL)
                if not text_match:
                    continue
                text = text_match.group(1).strip()
                unlabeled_texts.append(text)
                unlabeled_labels.append(-1)
            else:
                text, _ = text_label
                unlabeled_texts.append(text)
                unlabeled_labels.append(-1)
    return labeled_texts, labels, unlabeled_texts, unlabeled_labels


class AmazonSentimentDataset(Dataset):
    """PyTorch dataset for tokenized Amazon reviews."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


###############################################################################
# Domain‑adversarial neural network (DANN) implementation
###############################################################################

class GradientReversalFunction(torch.autograd.Function):
    """Autograd function implementing the gradient reversal layer.

    During the forward pass this function acts as the identity.  During the
    backward pass it multiplies the gradient by −λ, effectively reversing the
    gradient flow.  This encourages the encoder to learn representations that
    fool the domain classifier, thereby making them domain invariant.
    """

    @staticmethod
    def forward(ctx, input_, lambda_):
        ctx.lambda_ = lambda_
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, input_):
        return GradientReversalFunction.apply(input_, self.lambda_)


class DANNModel(nn.Module):
    """Domain‑Adversarial Neural Network for sentiment classification.

    The model consists of a pre‑trained language model encoder (e.g., BERT), a
    sentiment classifier and a domain classifier.  During training the encoder
    attempts to minimize the sentiment classification loss while
    simultaneously maximizing the domain classification loss via the gradient
    reversal layer.  This adversarial setup encourages the encoder to
    produce representations that are both task‑relevant and domain invariant.
    """

    def __init__(self, encoder_name: str = "bert-base-uncased", num_labels: int = 2, lambda_da: float = 1.0):
        super().__init__()
        self.lambda_da = lambda_da
        # Load pre‑trained encoder
        self.encoder_config = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name, config=self.encoder_config)
        hidden_size = self.encoder_config.hidden_size
        # Sentiment classifier: simple linear layer
        self.classifier = nn.Linear(hidden_size, num_labels)
        # Domain classifier
        self.grl = GradientReversalLayer(lambda_da)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, input_ids, attention_mask=None, domain_labels=None):
        # Compute encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        # Sentiment logits
        sentiment_logits = self.classifier(pooled_output)
        # Domain logits via gradient reversal
        reversed_feature = self.grl(pooled_output)
        domain_logits = self.domain_classifier(reversed_feature)
        return sentiment_logits, domain_logits


def train_dann(
    model: DANNModel,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int = 3,
    lambda_da: float = 1.0,
) -> None:
    """Train the domain‑adversarial model.

    Args:
        model: The DANN model to train.
        source_loader: DataLoader providing labeled source domain batches.
        target_loader: DataLoader providing unlabeled target domain batches.  The
            labels for these examples can be dummy (e.g., -1) since they are not
            used for sentiment loss.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        device: Computation device (CPU or GPU).
        epochs: Number of training epochs.
        lambda_da: Weight for domain adversarial loss.
    """
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Create iterators over source and target loaders
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        # Ensure we iterate over the larger of the two datasets
        num_batches = max(len(source_loader), len(target_loader))
        for _ in range(num_batches):
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_batch = next(source_iter)
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            # Prepare inputs
            input_ids = torch.cat([source_batch["input_ids"], target_batch["input_ids"]], dim=0).to(device)
            attention_mask = torch.cat([source_batch["attention_mask"], target_batch["attention_mask"]], dim=0).to(device)
            # Sentiment labels (source: real labels, target: dummy zeros)
            sentiment_labels = torch.cat([
                source_batch["labels"],
                torch.zeros_like(target_batch["labels"]),
            ], dim=0).to(device)
            # Domain labels: 0 for source, 1 for target
            domain_labels = torch.cat([
                torch.zeros(len(source_batch["labels"]), dtype=torch.long),
                torch.ones(len(target_batch["labels"]), dtype=torch.long),
            ], dim=0).to(device)
            optimizer.zero_grad()
            sentiment_logits, domain_logits = model(input_ids, attention_mask)
            # Compute losses
            loss_senti = loss_fn(sentiment_logits[: len(source_batch["labels"]), :], source_batch["labels"].to(device))
            loss_domain = loss_fn(domain_logits, domain_labels)
            loss = loss_senti + lambda_da * loss_domain
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs} – Loss: {avg_loss:.4f}")


def evaluate_model(model: DANNModel, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the sentiment accuracy of the model on a labeled dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            sentiment_logits, _ = model(input_ids, attention_mask)
            predictions = torch.argmax(sentiment_logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


###############################################################################
# Knowledge distillation
###############################################################################

def distillation_loss(student_logits, teacher_logits, temperature: float = 2.0) -> torch.Tensor:
    """Compute the soft cross‑entropy between student and teacher distributions.

    Args:
        student_logits: Logits output by the student model.
        teacher_logits: Logits output by the teacher model.
        temperature: Temperature to soften distributions.

    Returns:
        Kullback–Leibler divergence between softened distributions.
    """
    # Soften probabilities with temperature
    student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return loss


def train_student(
    teacher_model: DANNModel,
    student_model: AutoModelForSequenceClassification,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int = 3,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> None:
    """Train a student model using knowledge distillation.

    The student learns from the teacher’s soft outputs on both the labeled
    source data and the unlabeled target data.  It also receives supervised
    signals from the labeled source examples.  The loss is a weighted
    combination of the supervised cross‑entropy and the distillation loss.

    Args:
        teacher_model: Trained DANN model acting as teacher.
        student_model: A compact sequence classifier (e.g., DistilBERT).
        source_loader: DataLoader for labeled source domain.
        target_loader: DataLoader for unlabeled target domain.
        optimizer: Optimizer for the student model.
        scheduler: Learning rate scheduler.
        device: Computation device.
        epochs: Number of training epochs.
        alpha: Weight for the distillation loss (between 0 and 1).  The
            supervised loss weight becomes (1 − alpha).
        temperature: Softening temperature for distillation.
    """
    teacher_model.to(device)
    teacher_model.eval()
    student_model.to(device)
    student_model.train()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        num_batches = max(len(source_loader), len(target_loader))
        for _ in range(num_batches):
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_batch = next(source_iter)
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            # Combine source and target texts
            all_input_ids = torch.cat([source_batch["input_ids"], target_batch["input_ids"]], dim=0).to(device)
            all_attention_mask = torch.cat([source_batch["attention_mask"], target_batch["attention_mask"]], dim=0).to(device)
            # Forward through teacher
            with torch.no_grad():
                teacher_logits, _ = teacher_model(all_input_ids, all_attention_mask)
            # Forward through student
            student_outputs = student_model(input_ids=all_input_ids, attention_mask=all_attention_mask)
            student_logits = student_outputs.logits
            # Distillation loss on both source and target
            loss_kd = distillation_loss(student_logits, teacher_logits, temperature)
            # Supervised loss on the source part only
            supervised_logits = student_logits[: len(source_batch["labels"]), :]
            supervised_labels = source_batch["labels"].to(device)
            loss_sup = loss_fn(supervised_logits, supervised_labels)
            # Combine losses
            loss = alpha * loss_kd + (1.0 - alpha) * loss_sup
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / num_batches
        print(f"Student Epoch {epoch + 1}/{epochs} – Loss: {avg_loss:.4f}")


def evaluate_student(model: AutoModelForSequenceClassification, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluate a distilled student model on labeled data."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


###############################################################################
# Main execution
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Domain adaptation and knowledge distillation for Amazon reviews")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store/download the dataset")
    parser.add_argument("--source_domain", type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"], help="Source domain for training")
    parser.add_argument("--target_domain", type=str, default="electronics", choices=["books", "dvd", "electronics", "kitchen"], help="Target domain for adaptation")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased", help="Name of the large pre‑trained model for the DANN teacher")
    parser.add_argument("--student_model", type=str, default="distilbert-base-uncased", help="Name of the compact model for distillation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs_da", type=int, default=3, help="Number of epochs for domain adversarial training")
    parser.add_argument("--epochs_kd", type=int, default=3, help="Number of epochs for knowledge distillation training")
    parser.add_argument("--lambda_da", type=float, default=0.1, help="Weight for domain adversarial loss")
    parser.add_argument("--alpha_kd", type=float, default=0.5, help="Weight for distillation loss in KD")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    # Step 0: Download and extract dataset
    archive = download_mdsd(args.data_dir)
    extract_dir = os.path.join(args.data_dir, "mdsd")
    extract_mdsd(archive, extract_dir)
    domain_map = {
        "books": "books",
        "dvd": "dvd",
        "electronics": "electronics",
        "kitchen": "kitchen",
    }
    source_dir = os.path.join(extract_dir, domain_map[args.source_domain])
    target_dir = os.path.join(extract_dir, domain_map[args.target_domain])
    # Load data for source and target domains
    source_texts, source_labels, _, _ = load_domain_dataset(source_dir)
    target_texts, _, target_unlabeled_texts, _ = load_domain_dataset(target_dir)
    # Create train/dev splits; we reserve 10% of labeled source for validation
    n_source = len(source_texts)
    val_size = max(1, int(0.1 * n_source))
    train_texts = source_texts[:-val_size]
    train_labels = source_labels[:-val_size]
    val_texts = source_texts[-val_size:]
    val_labels = source_labels[-val_size:]
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    # Build datasets
    source_train_dataset = AmazonSentimentDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    source_val_dataset = AmazonSentimentDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    # For unsupervised adaptation we treat unlabeled target texts with dummy label 0
    target_train_dataset = AmazonSentimentDataset(target_unlabeled_texts, [-1] * len(target_unlabeled_texts), tokenizer, max_length=args.max_length)
    target_val_dataset = AmazonSentimentDataset(target_texts, [0] * len(target_texts), tokenizer, max_length=args.max_length)
    # DataLoaders
    source_train_loader = DataLoader(source_train_dataset, sampler=RandomSampler(source_train_dataset), batch_size=args.batch_size)
    source_val_loader = DataLoader(source_val_dataset, sampler=SequentialSampler(source_val_dataset), batch_size=args.batch_size)
    target_train_loader = DataLoader(target_train_dataset, sampler=RandomSampler(target_train_dataset), batch_size=args.batch_size)
    target_val_loader = DataLoader(target_val_dataset, sampler=SequentialSampler(target_val_dataset), batch_size=args.batch_size)
    # Step 1: Train domain adversarial teacher model
    print("=== Stage 1: Domain Adversarial Training ===")
    teacher_model = DANNModel(encoder_name=args.pretrained_model, num_labels=2, lambda_da=args.lambda_da)
    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=5e-5)
    total_steps = args.epochs_da * max(len(source_train_loader), len(target_train_loader))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    train_dann(teacher_model, source_train_loader, target_train_loader, optimizer, scheduler, device=torch.device(args.device), epochs=args.epochs_da, lambda_da=args.lambda_da)
    # Evaluate teacher on source validation and target validation
    val_source_acc = evaluate_model(teacher_model, source_val_loader, device=torch.device(args.device))
    val_target_acc = evaluate_model(teacher_model, target_val_loader, device=torch.device(args.device))
    print(f"Teacher validation accuracy on source ({args.source_domain}): {val_source_acc:.4f}")
    print(f"Teacher validation accuracy on target ({args.target_domain}): {val_target_acc:.4f}")
    # Step 2: Knowledge distillation into a student model
    print("=== Stage 2: Knowledge Distillation ===")
    student_model = AutoModelForSequenceClassification.from_pretrained(args.student_model, num_labels=2)
    optimizer_student = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    total_steps_kd = args.epochs_kd * max(len(source_train_loader), len(target_train_loader))
    scheduler_student = get_linear_schedule_with_warmup(optimizer_student, num_warmup_steps=0, num_training_steps=total_steps_kd)
    train_student(
        teacher_model=teacher_model,
        student_model=student_model,
        source_loader=source_train_loader,
        target_loader=target_train_loader,
        optimizer=optimizer_student,
        scheduler=scheduler_student,
        device=torch.device(args.device),
        epochs=args.epochs_kd,
        alpha=args.alpha_kd,
        temperature=args.temperature,
    )
    # Evaluate student
    student_source_acc = evaluate_student(student_model, source_val_loader, device=torch.device(args.device))
    student_target_acc = evaluate_student(student_model, target_val_loader, device=torch.device(args.device))
    print(f"Student validation accuracy on source ({args.source_domain}): {student_source_acc:.4f}")
    print(f"Student validation accuracy on target ({args.target_domain}): {student_target_acc:.4f}")


if __name__ == "__main__":
    main()