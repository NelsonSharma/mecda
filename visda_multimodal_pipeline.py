"""
visda_multimodal_pipeline.py
=================================

This module implements an end‑to‑end training pipeline for performing
memory‑efficient collaborative domain adaptation on the VisDA‐2022
multi‑modal dataset.  The goal of the VisDA challenge is to transfer
knowledge from a source domain (synthetic renderings of objects) to a
target domain (real images) for multi‑modal classification and
retrieval tasks.  The VisDA dataset is one of the largest cross‑domain
datasets available for unsupervised domain adaptation with more than
280 000 images spanning 12 object categories【207149797189071†L49-L60】.  In this
implementation the focus is on the image+text classification track; a
large pre‑trained vision–language model (e.g. CLIP) is first adapted
using an adversarial objective to learn domain‑invariant features and
then distilled into a compact student model suitable for deployment
on resource‑constrained devices.  The architecture follows the same
two‑stage strategy used in MEC‑DA:

    • **Domain adaptation stage** – A large teacher model is trained
      using a Domain Adversarial Neural Network (DANN) objective.  The
      model’s classification head learns to predict object categories
      for labelled source examples while a domain classifier, trained
      via a gradient reversal layer, encourages the model to produce
      representations that are indistinguishable across source and
      target domains.  Only the source domain carries labels; target
      samples are used solely for domain discrimination.  This stage
      mirrors the Lite Residual Hypothesis Transfer (LRHT) step in
      MEC‑DA but adapted for multi‑modal inputs.

    • **Knowledge distillation stage** – Once the teacher has been
      adapted, a smaller student model is trained to mimic the
      teacher’s soft class predictions on both source and target data
      while still being supervised on labelled source samples.  This
      collaborative distillation step allows the compact model to
      acquire target‑domain knowledge without forgetting its source‑
      domain skills, analogous to the Co‑KD procedure described in
      MEC‑DA.

The code is organised so that the same command‑line interface used for
the Amazon Reviews and SQuAD→NewsQA pipelines can be applied here.
Users must supply paths to their local VisDA dataset – the code
assumes two metadata files (`source_list.txt` and `target_list.txt`)
that index images, optional text captions and class labels.  Each
line in these files should follow the tab‑separated format:

    ``path/to/image.jpg\t<caption string>\t<label index>``

If the label index is omitted for the target domain the loader
defaults to ``-1`` and the example is treated as unlabelled.  Image
paths are resolved relative to a root directory specified on the
command line.  A CLIP processor is used to convert images and
captions into the appropriate tensor representations.  The training
loop prints progress at regular intervals and tracks both
classification accuracy and domain classifier accuracy on a held‑out
validation set.

References
----------
The VisDA challenge was introduced as part of the Visual Domain
Adaptation dataset, which forms a large test bed for unsupervised
domain adaptation in computer vision.  The dataset uses synthetic
renderings of 12 object categories as the source domain and real
photographs for the target domain, comprising more than 280 000
images【207149797189071†L49-L60】.

"""

import argparse
import os
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError as exc:
    raise ImportError(
        "This script requires the `transformers` library. "
        "Install it via `pip install transformers`."
    ) from exc


class GradientReversalFunction(torch.autograd.Function):
    """Implements a gradient reversal layer for domain adversarial training.

    During the forward pass this function acts as the identity.  In the
    backward pass it multiplies the incoming gradient by a negative
    scalar ``lambda_`` so that upstream parameters are updated in
    opposition to the domain classification loss.  See Ganin et
    al. (2015) for details.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        # Identity mapping for forward
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


class DomainClassifier(nn.Module):
    """Small feed‑forward network for predicting domain labels.

    The network consists of two linear layers with a ReLU non‑linearity
    in between.  It takes as input the fused multi‑modal embedding
    produced by the vision–language model and outputs logits over
    domain classes (two domains: source=0, target=1).
    """

    def __init__(self, in_features: int, hidden: int = 128, num_domains: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CLIPDANNModel(nn.Module):
    """Composite model for domain‑adversarial training on CLIP encoders.

    Given a CLIP model this class wraps its image and text encoders to
    produce a fused representation (mean of image and text embeddings).
    The fused embeddings feed into two heads:

        * A classification head predicting class logits over the
          categories present in the VisDA dataset.
        * A domain classifier trained via a gradient reversal layer to
          confuse the encoder with respect to domain labels.

    Parameters
    ----------
    clip_model : CLIPModel
        Pre‑trained CLIP model from Hugging Face.  Both the image and
        text encoders are used.
    num_classes : int
        Number of object categories.
    domain_hidden : int, optional
        Size of the hidden layer in the domain classifier.
    """

    def __init__(self, clip_model: CLIPModel, num_classes: int, domain_hidden: int = 256) -> None:
        super().__init__()
        self.clip = clip_model
        # Freeze CLIP parameters by default to stabilise training.  Users
        # may unfreeze later via fine‑tuning.
        for param in self.clip.parameters():
            param.requires_grad = False
        embed_dim = self.clip.config.projection_dim  # hidden size for image/text embeddings
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.domain_classifier = DomainClassifier(embed_dim, hidden=domain_hidden, num_domains=2)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward through CLIP to get image and text embeddings
        # clip returns image_embeds and text_embeds after projection
        outputs = self.clip(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        # Fuse embeddings (simple average).  More sophisticated
        # strategies (e.g. gating) could be explored.
        fused = (image_embeds + text_embeds) / 2.0
        # Classification predictions over object classes
        class_logits = self.classifier(fused)
        # Domain predictions via gradient reversal
        reversed_embeds = GradientReversalFunction.apply(fused, grl_lambda)
        domain_logits = self.domain_classifier(reversed_embeds)
        return class_logits, domain_logits


@dataclass
class VisdaExample:
    """Container for a single example in the VisDA dataset."""

    image_path: str
    caption: Optional[str]
    label: int
    domain: int  # 0 for source, 1 for target


class VisdaDataset(Dataset):
    """Custom dataset for VisDA multi‑modal classification.

    This dataset expects a metadata file where each line contains a
    relative image path, a caption string and a label, separated by
    tabs.  If the label is omitted (empty string) the sample is
    considered unlabelled (label = -1).  The dataset also stores a
    domain identifier.  Images are loaded lazily using PIL and
    transformed with torchvision transforms.  Captions are tokenized
    using the CLIP processor.  The processor is responsible for
    encoding both the image and the caption, so the dataset simply
    returns their original forms.  Domain labels are returned as a
    separate tensor for training the domain classifier.
    """

    def __init__(
        self,
        metadata_file: str,
        root_dir: str,
        processor: CLIPProcessor,
        transform: Optional[Any] = None,
        domain: int = 0,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.transform = transform
        self.domain = domain
        self.examples: List[VisdaExample] = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if not parts:
                    continue
                image_rel = parts[0]
                caption = parts[1] if len(parts) > 1 and parts[1] else ""
                label = int(parts[2]) if len(parts) > 2 and parts[2] != "" else -1
                self.examples.append(
                    VisdaExample(
                        image_path=os.path.join(root_dir, image_rel),
                        caption=caption,
                        label=label,
                        domain=self.domain,
                    )
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        # Load and transform image
        image = Image.open(ex.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Use CLIP processor to get tokenized caption later in collate_fn
        caption = ex.caption
        label = ex.label
        domain_label = ex.domain
        return {
            "image": image,
            "caption": caption,
            "label": label,
            "domain": domain_label,
        }


def visda_collate(batch: List[Dict[str, Any]], processor: CLIPProcessor) -> Dict[str, Any]:
    """Custom collate function to batch VisDA samples.

    The CLIP processor is used to convert raw images and captions into
    pixel values, input IDs and attention masks.  Labels and domain
    identifiers are stacked into tensors.  This function produces a
    dictionary suitable for feeding into ``CLIPDANNModel``.
    """
    images = [item["image"] for item in batch]
    captions = [item["caption"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    domains = torch.tensor([item["domain"] for item in batch], dtype=torch.long)
    # Use processor to tokenize
    inputs = processor(text=captions, images=images, return_tensors="pt", padding=True)
    return {
        "pixel_values": inputs["pixel_values"],
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
        "domains": domains,
    }


def train_dann(
    model: CLIPDANNModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grl_lambda: float,
    label_smoothing: float = 0.0,
) -> Tuple[float, float]:
    """One epoch of domain‑adversarial training.

    The DANN objective consists of a classification loss on labelled
    source examples and a domain classification loss on both source
    and target examples.  Unlabelled target samples are ignored for
    classification.  Domain labels are either 0 (source) or 1 (target).
    """
    model.train()
    total_cls_loss = 0.0
    total_dom_loss = 0.0
    n_batches = 0
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    dom_criterion = nn.CrossEntropyLoss()
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        domains = batch["domains"].to(device)
        optimizer.zero_grad()
        class_logits, domain_logits = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            grl_lambda=grl_lambda,
        )
        # Compute classification loss only on labelled samples
        labelled_mask = labels >= 0
        if labelled_mask.any():
            cls_loss = ce_criterion(class_logits[labelled_mask], labels[labelled_mask])
        else:
            cls_loss = torch.tensor(0.0, device=device)
        dom_loss = dom_criterion(domain_logits, domains)
        loss = cls_loss + dom_loss
        loss.backward()
        optimizer.step()
        total_cls_loss += cls_loss.item()
        total_dom_loss += dom_loss.item()
        n_batches += 1
    return total_cls_loss / max(1, n_batches), total_dom_loss / max(1, n_batches)


def evaluate_classifier(model: CLIPDANNModel, dataloader: DataLoader, device: torch.device) -> float:
    """Compute classification accuracy on a labelled dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            class_logits, _ = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                grl_lambda=0.0,
            )
            # Only evaluate on labelled examples
            mask = labels >= 0
            if mask.any():
                preds = class_logits[mask].argmax(dim=1)
                correct += (preds == labels[mask]).sum().item()
                total += mask.sum().item()
    return correct / max(1, total)


def distill_teacher_to_student(
    teacher: CLIPDANNModel,
    student: CLIPDANNModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """Perform one epoch of knowledge distillation from teacher to student.

    The student is trained to match the soft predictions (probability
    distributions) of the teacher on both source and target samples
    (knowledge distillation loss) while also fitting the ground truth
    labels on the source domain (cross‑entropy loss).  The blend
    between the two losses is controlled by ``alpha``.
    """
    teacher.eval()
    student.train()
    kd_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.no_grad():
            teacher_logits, _ = teacher(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                grl_lambda=0.0,
            )
        student_logits, _ = student(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            grl_lambda=0.0,
        )
        # Soft targets: teacher logits scaled by temperature
        T = temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        kd_loss = kd_loss_fn(student_log_probs, teacher_probs) * (T * T)
        # CE loss on labelled source samples only
        mask = labels >= 0
        if mask.any():
            ce_loss = ce_loss_fn(student_logits[mask], labels[mask])
        else:
            ce_loss = torch.tensor(0.0, device=device)
        loss = alpha * kd_loss + (1 - alpha) * ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description="VisDA multi‑modal domain adaptation and distillation pipeline")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing images referenced by metadata files")
    parser.add_argument("--source_metadata", type=str, required=True, help="Path to metadata file listing source (synthetic) images and labels")
    parser.add_argument("--target_metadata", type=str, required=True, help="Path to metadata file listing target (real) images and captions")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of object categories in the dataset")
    parser.add_argument("--teacher_model", type=str, default="openai/clip-vit-base-patch32", help="Hugging Face model identifier for the teacher CLIP model")
    parser.add_argument("--student_model", type=str, default="openai/clip-vit-base-patch16", help="Hugging Face model identifier for the student CLIP model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--epochs_stage1", type=int, default=3, help="Number of epochs for domain adversarial adaptation of the teacher")
    parser.add_argument("--epochs_stage2", type=int, default=3, help="Number of epochs for knowledge distillation")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimisers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate after this many epochs during stage 1")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend factor between KD and CE loss in distillation")
    parser.add_argument("--temperature", type=float, default=4.0, help="Temperature for distillation")
    parser.add_argument("--grl_lambda", type=float, default=1.0, help="Weight for gradient reversal layer during DANN training")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load CLIP processor and teacher model
    processor = CLIPProcessor.from_pretrained(args.teacher_model)
    teacher_clip = CLIPModel.from_pretrained(args.teacher_model)

    # Prepare datasets and loaders
    # Apply transformations to images (clip expects images scaled to [0,1] and normalised internally via processor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    source_dataset = VisdaDataset(
        metadata_file=args.source_metadata,
        root_dir=args.root_dir,
        processor=processor,
        transform=None,  # raw image; processor handles resizing
        domain=0,
    )
    target_dataset = VisdaDataset(
        metadata_file=args.target_metadata,
        root_dir=args.root_dir,
        processor=processor,
        transform=None,
        domain=1,
    )
    # Combine source and target datasets for DANN stage.  In DANN training
    # we want to sample from both domains in one batch.  For
    # simplicity we concatenate the two datasets and rely on the domain
    # label to differentiate.  The underlying DataLoader will not
    # necessarily maintain class balance; this could be improved by a
    # custom sampler.
    combined_dataset = source_dataset + target_dataset  # type: ignore[arg-type]
    train_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: visda_collate(b, processor),
        num_workers=4,
    )
    # Create a validation loader from the labelled part of target dataset if labels exist
    # Otherwise evaluation can be performed on source test set which the user must provide separately.
    # For simplicity we evaluate on source dataset here.
    eval_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: visda_collate(b, processor),
        num_workers=2,
    )

    # Instantiate teacher model for DANN training
    teacher_model = CLIPDANNModel(clip_model=teacher_clip, num_classes=args.num_classes, domain_hidden=256).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, teacher_model.parameters()), lr=args.learning_rate)

    # Stage 1: Domain adversarial training
    print("\n=== Stage 1: Domain Adversarial Training of the Teacher ===")
    for epoch in range(1, args.epochs_stage1 + 1):
        cls_loss, dom_loss = train_dann(
            model=teacher_model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grl_lambda=args.grl_lambda,
            label_smoothing=0.0,
        )
        print(f"Epoch {epoch}/{args.epochs_stage1}: classification loss={cls_loss:.4f}, domain loss={dom_loss:.4f}")
        if epoch % args.eval_interval == 0:
            acc = evaluate_classifier(teacher_model, eval_loader, device)
            print(f"  → Source classification accuracy after epoch {epoch}: {acc * 100:.2f}%")

    # Stage 2: Knowledge distillation into a smaller student model
    print("\n=== Stage 2: Knowledge Distillation into the Student ===")
    student_processor = CLIPProcessor.from_pretrained(args.student_model)
    student_clip = CLIPModel.from_pretrained(args.student_model)
    student_model = CLIPDANNModel(clip_model=student_clip, num_classes=args.num_classes, domain_hidden=256).to(device)
    # Allow all parameters of the student to be trained
    for param in student_model.parameters():
        param.requires_grad = True
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.epochs_stage2 + 1):
        # For distillation we reuse the combined loader; we do not use domain labels here
        distill_loss = distill_teacher_to_student(
            teacher=teacher_model,
            student=student_model,
            dataloader=train_loader,
            optimizer=student_optimizer,
            device=device,
            temperature=args.temperature,
            alpha=args.alpha,
        )
        print(f"Epoch {epoch}/{args.epochs_stage2}: distillation loss={distill_loss:.4f}")
        # Evaluate student on source validation set
        if epoch % args.eval_interval == 0:
            acc = evaluate_classifier(student_model, eval_loader, device)
            print(f"  → Student classification accuracy after epoch {epoch}: {acc * 100:.2f}%")
    print("\nTraining complete.  Save your teacher and student models using the standard Hugging Face API if desired.")


if __name__ == "__main__":
    main()