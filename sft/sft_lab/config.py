"""Typed and validated configurations for SFT and GRPO experiments."""

import os
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class SFTConfig(BaseModel):
    """Validation schema for SFT configuration."""

    model_id: str = Field("Qwen/Qwen3.8-27B", description="HF Model ID")
    tokenizer_id: str = Field("Qwen/Qwen3.8-27B", description="HF Tokenizer ID")
    seed: int = Field(42, ge=0, description="Random seed")
    max_seq_length: int = Field(8192, gt=0, description="Maximum sequence length")
    dataset_name_or_path: str = Field(..., description="Train dataset path or HF dataset name")
    eval_dataset_path: Optional[str] = Field(None, description="Validation dataset path")
    train_split: str = Field("train", description="Dataset train split name")
    eval_split: str = Field("eval", description="Dataset eval split name")
    assistant_only_loss: bool = Field(True, description="Whether to mask non-assistant tokens in loss")
    batch_size: int = Field(1, gt=0, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(16, gt=0, description="Gradient accumulation steps")
    learning_rate: float = Field(2e-5, gt=0.0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0.0, description="Weight decay")
    num_epochs: int = Field(3, gt=0, description="Number of training epochs")
    warmup_ratio: float = Field(0.05, ge=0.0, le=1.0, description="Warmup ratio")
    output_dir: str = Field("artifacts/checkpoints/sft", description="Output directory for checkpoints")
    dtype: str = Field("bfloat16", description="Model dtype (float32, float16, bfloat16)")
    quantization: Optional[str] = Field(None, description="Quantization mode (4bit, 8bit, fp8)")
    grad_clip: float = Field(1.0, ge=0.0, description="Gradient clipping norm")
    eval_steps: int = Field(25, ge=1, description="Evaluation frequency in steps")
    save_steps: int = Field(25, ge=1, description="Checkpoint saving frequency in steps")
    logging_steps: int = Field(1, ge=1, description="Logging frequency in steps")
    supervise_cot: bool = Field(True, description="Supervise chain of thought")
    supervise_tools: bool = Field(True, description="Supervise tool calls")
    allow_zero_supervised_tokens: bool = Field(False, description="Allow examples with 0 supervised tokens")

    @model_validator(mode="after")
    def validate_config(self) -> "SFTConfig":
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SFTConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, yaml_path: str) -> None:
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f)
