#!/usr/bin/env python3
"""Train a local Llama 3 LoRA adapter on pre-deadline FPL forecast examples.

Unsloth needs a supported NVIDIA GPU under Linux/WSL2 (not native Windows).
Transformers/TRL is an alternative backend; --no-load-in-4bit permits CPU
execution but an 8B model needs substantial RAM and is very slow. Install the
correct PyTorch/CUDA build separately; this script never installs dependencies.
Hugging Face gated Llama models require accepted access terms and HF login.
The Transformers backend takes a base model, not an existing adapter directory:
retrain weekly on the accumulated dataset, or resume an unchanged Trainer run
with --resume-from-checkpoint. Unsloth can warm-start from a local adapter path.
"""

import argparse
import json
import math
from pathlib import Path
import sys

from training.data import chronological_split, load_examples
from training.prompts import SYSTEM_PROMPT, user_prompt


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data") / "training.jsonl")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="HF Llama 3 model ID or local model/Unsloth adapter path")
    parser.add_argument("--output", type=Path, default=Path("models") / "fpl-lora")
    parser.add_argument("--backend", choices=("unsloth", "transformers"), default="unsloth")
    parser.add_argument("--max-seq-length", type=positive_int, default=8192)
    parser.add_argument("--rank", type=positive_int, default=16)
    parser.add_argument("--lora-alpha", type=positive_int, default=32)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--gradient-accumulation", type=positive_int, default=8)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1,
                        help="Newest gameweeks held out; never shuffled across the time boundary")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", type=Path,
                        help="Trainer checkpoint with optimizer state from the same data/model configuration")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate JSONL and chronological split without importing ML libraries")
    args = parser.parse_args(argv)
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be positive and finite")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if not math.isfinite(args.validation_fraction) or not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        examples = load_examples(args.data)
        train_rows, eval_rows = chronological_split(examples, args.validation_fraction)
        print(f"Validated {len(examples)} gameweeks: {len(train_rows)} training, {len(eval_rows)} validation")
        if args.validate_only:
            return 0
        if args.resume_from_checkpoint and not args.resume_from_checkpoint.is_dir():
            raise ValueError("--resume-from-checkpoint must point to an existing Trainer checkpoint directory")
        if args.backend == "transformers" and (Path(args.model) / "adapter_config.json").is_file():
            raise ValueError(
                "--backend transformers requires a base model, not a LoRA adapter directory; "
                "retrain on the accumulated dataset, or use --resume-from-checkpoint "
                "for an unchanged Trainer run"
            )
        if args.backend == "unsloth":
            if sys.platform == "win32":
                raise RuntimeError("Use Unsloth under Linux/WSL2, or select --backend transformers.")
            # Import before transformers/TRL so Unsloth can apply its supported patches.
            from unsloth import FastLanguageModel

        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        cuda = torch.cuda.is_available()
        if (args.load_in_4bit or args.backend == "unsloth") and not cuda:
            raise RuntimeError("Unsloth/4-bit training requires a supported CUDA GPU. No CUDA packages are auto-installed.")
        bf16 = cuda and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16 if cuda else torch.float32
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if args.backend == "unsloth":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model, max_seq_length=args.max_seq_length,
                dtype=dtype, load_in_4bit=args.load_in_4bit,
            )
            if not getattr(model, "peft_config", None):
                model = FastLanguageModel.get_peft_model(
                    model, r=args.rank, target_modules=targets, lora_alpha=args.lora_alpha,
                    lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
                    random_state=args.seed,
                )
            else:
                FastLanguageModel.for_training(model)
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            kwargs = {"torch_dtype": dtype}
            if args.load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = {"": torch.cuda.current_device()}
            model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
            if args.load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, LoraConfig(
                r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=0,
                target_modules=targets, bias="none", task_type="CAUSAL_LM",
            ))
        if not tokenizer.chat_template or not tokenizer.eos_token:
            raise ValueError("The chosen model must supply a Llama 3 instruction chat template and EOS token")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.use_cache = False

        def dataset(rows):
            formatted = []
            for row in rows:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(row["instruction"], row["input"])},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                answer = json.dumps(row["output"], ensure_ascii=False, sort_keys=True, allow_nan=False)
                full = tokenizer.apply_chat_template(
                    messages + [{"role": "assistant", "content": answer}],
                    tokenize=False, add_generation_prompt=False,
                )
                if not full.startswith(prompt):
                    raise ValueError("Chat template must preserve the generation prompt prefix")
                completion = full[len(prompt):]
                if len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]) > args.max_seq_length:
                    raise ValueError(
                        f"GW{row['input']['gameweek']} exceeds --max-seq-length; increase it rather than truncating labels"
                    )
                formatted.append({"prompt": prompt, "completion": completion})
            return Dataset.from_list(formatted)

        train_dataset = dataset(train_rows)
        eval_dataset = dataset(eval_rows) if eval_rows else None
        config = SFTConfig(
            output_dir=str(args.output),
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=cuda and not bf16,
            bf16=bf16,
            optim="adamw_torch",
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch" if eval_rows else "no",
            completion_only_loss=True,
            packing=False,
            gradient_checkpointing=True,
            seed=args.seed,
            data_seed=args.seed,
            report_to="none",
            dataloader_num_workers=0,
        )
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, args=config,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
        )
        model.print_trainable_parameters()
        result = trainer.train(
            resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
        trainer.save_model(str(args.output))
        tokenizer.save_pretrained(str(args.output))
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        manifest = {
            "base_model": args.model, "backend": args.backend, "data": str(args.data.resolve()),
            "system_prompt": SYSTEM_PROMPT,
            "train_gameweeks": [[row["input"]["season"], row["input"]["gameweek"]] for row in train_rows],
            "validation_gameweeks": [[row["input"]["season"], row["input"]["gameweek"]] for row in eval_rows],
            "label_type": "realized_individual_points",
            "supervision": "assistant completion only; no outcome metadata in model input",
        }
        (args.output / "fpl_training.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved LoRA adapter and tokenizer to {args.output}")
        return 0
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
