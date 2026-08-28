"""Command parser and dispatch implementation."""

from __future__ import annotations

import json
import platform

import torch

from ..config import MiniScaleConfig
from ..data.agent import audit_agent_jsonl, save_agent_data_audit
from ..data.preference_audit import audit_dpo_jsonl, save_dpo_data_audit
from ..data.pretrain_audit import audit_pretrain_jsonl, save_data_audit
from ..data.rl import audit_rl_jsonl, save_rl_data_audit
from ..data.sft_audit import audit_sft_jsonl, save_sft_data_audit
from ..data.sft_prepare import prepare_sft_jsonl
from ..evaluation import evaluate_rl_checkpoints
from ..inference import GenerationOptions, generate_from_checkpoint
from ..model import MiniScaleForCausalLM
from ..pipeline import run_training_pipeline
from ..tokenizer import load_tokenizer, train_sentencepiece
from ..training import (
    AgentRLOptions, DPOOptions, GRPOOptions, PretrainOptions, SFTOptions,
    run_agent_grpo_jsonl, run_dpo_jsonl, run_grpo_jsonl, run_pretrain_jsonl, run_sft_jsonl,
)
from ..training.core.checkpoint import load_checkpoint
from ..training.core.runtime import seed_everything
from .parser import build_parser


def environment_report() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "doctor":
        result = environment_report()
    elif arguments.command == "pipeline":
        result = run_training_pipeline(arguments.output, device=arguments.device)
    elif arguments.command == "evaluate":
        result = evaluate_rl_checkpoints(
            arguments.checkpoint,
            arguments.data,
            arguments.tokenizer,
            kind=arguments.kind,
            validation_fraction=arguments.validation_fraction,
            prompts=arguments.prompts,
            max_new_tokens=arguments.max_new_tokens,
            max_turns=arguments.max_turns,
            precision=arguments.precision,
            seed=arguments.seed,
            device=arguments.device,
            output_path=arguments.output,
        )
    elif arguments.command == "generate":
        result = generate_from_checkpoint(
            arguments.checkpoint,
            arguments.prompt,
            GenerationOptions(
                max_new_tokens=arguments.max_new_tokens,
                temperature=arguments.temperature,
                top_k=arguments.top_k or None,
                device=arguments.device,
                system_prompt=arguments.system_prompt,
                tokenizer_path=arguments.tokenizer,
                raw_prompt=arguments.raw_prompt,
                calculator=arguments.calculator,
                max_turns=arguments.max_turns,
            ),
        )
        if arguments.raw:
            print(result["response"])
            return
    elif arguments.command == "tokenize":
        tokenizer = load_tokenizer(arguments.tokenizer)
        token_ids = tokenizer.encode(arguments.text, bos=arguments.add_bos, eos=arguments.add_eos)
        decoded = tokenizer.decode(token_ids)
        result = {
            "tokenizer": str(arguments.tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "text": arguments.text,
            "token_ids": token_ids,
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "token_count": len(token_ids),
            "character_count": len(arguments.text),
            "characters_per_token": len(arguments.text) / max(len(token_ids), 1),
            "decoded": decoded,
            "round_trip": decoded == arguments.text,
        }
    elif arguments.command == "audit-pretrain-data":
        tokenizer = load_tokenizer(arguments.tokenizer)
        result = audit_pretrain_jsonl(
            arguments.data,
            tokenizer,
            validation_fraction=arguments.validation_fraction,
            sequence_length=arguments.sequence_length,
            tokenizer_batch_size=arguments.tokenizer_batch_size,
        )
        if arguments.output is not None:
            save_data_audit(result, arguments.output)
            result["report"] = str(arguments.output)
    elif arguments.command == "audit-sft-data":
        tokenizer = load_tokenizer(arguments.tokenizer)
        result = audit_sft_jsonl(
            arguments.data,
            tokenizer,
            max_length=arguments.max_length,
            min_context_tokens=arguments.min_context_tokens,
            target_mode=arguments.target_mode,
            validation_fraction=arguments.validation_fraction,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
            identity_patterns=arguments.identity_pattern,
        )
        if arguments.output is not None:
            save_sft_data_audit(result, arguments.output)
            result["report"] = str(arguments.output)
    elif arguments.command == "audit-dpo-data":
        tokenizer = load_tokenizer(arguments.tokenizer)
        result = audit_dpo_jsonl(
            arguments.data,
            tokenizer,
            max_length=arguments.max_length,
            min_context_tokens=arguments.min_context_tokens,
            target_mode=arguments.target_mode,
            validation_fraction=arguments.validation_fraction,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
        )
        if arguments.output is not None:
            save_dpo_data_audit(result, arguments.output)
            result["report"] = str(arguments.output)
    elif arguments.command == "audit-grpo-data":
        result = audit_rl_jsonl(
            arguments.data,
            validation_fraction=arguments.validation_fraction,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
        )
        if arguments.output is not None:
            save_rl_data_audit(result, arguments.output)
            result["report"] = str(arguments.output)
    elif arguments.command == "audit-agent-data":
        result = audit_agent_jsonl(
            arguments.data,
            validation_fraction=arguments.validation_fraction,
            seed=arguments.seed,
        )
        if arguments.output is not None:
            save_agent_data_audit(result, arguments.output)
            result["report"] = str(arguments.output)
    elif arguments.command == "prepare-sft-data":
        result = prepare_sft_jsonl(
            arguments.data,
            arguments.output,
            manifest_path=arguments.manifest,
            deduplicate_exact=arguments.deduplicate_exact,
            exclude_patterns=arguments.exclude_pattern,
        )
    elif arguments.command == "train-tokenizer":
        result = {"tokenizer": str(train_sentencepiece(
            arguments.data, arguments.output_prefix, vocab_size=arguments.vocab_size,
            input_sentence_size=arguments.input_sentences,
        ))}
    else:
        tokenizer = load_tokenizer(arguments.tokenizer)
        if arguments.command == "pretrain":
            # The training function receives an already-created model. Seed
            # before construction so --seed controls initial weights as well
            # as data order and later stochastic operations.
            seed_everything(arguments.seed)
            model = MiniScaleForCausalLM(MiniScaleConfig.small_64m(
                tokenizer.vocab_size,
                arguments.sequence_length,
                num_hidden_layers=arguments.num_hidden_layers,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ))
            result = run_pretrain_jsonl(model, tokenizer, arguments.data, arguments.output, PretrainOptions(
                steps=arguments.steps, batch_size=arguments.batch_size, sequence_length=arguments.sequence_length,
                learning_rate=arguments.learning_rate, min_learning_rate=arguments.min_learning_rate,
                weight_decay=arguments.weight_decay,
                adam_beta1=arguments.adam_beta1, adam_beta2=arguments.adam_beta2, adam_eps=arguments.adam_eps,
                grad_clip=arguments.grad_clip, seed=arguments.seed, precision=arguments.precision,
                warmup_steps=arguments.warmup_steps,
                gradient_accumulation_steps=arguments.gradient_accumulation,
                log_every=arguments.log_every, validation_every=arguments.validation_every,
                validation_batches=arguments.validation_batches, validation_fraction=arguments.validation_fraction,
                save_every=arguments.save_every,
                keep_last_checkpoints=arguments.keep_last,
                generation_every=arguments.generation_every,
                generation_max_new_tokens=arguments.generation_max_new_tokens,
                shuffle_buffer_size=arguments.shuffle_buffer_size,
                wandb_enabled=arguments.wandb, wandb_project=arguments.wandb_project,
                wandb_entity=arguments.wandb_entity, wandb_run_name=arguments.wandb_run_name,
                wandb_run_id=arguments.wandb_run_id, wandb_mode=arguments.wandb_mode,
                wandb_retry_every_steps=arguments.wandb_retry_every,
                resume_from=arguments.resume, allow_legacy_resume=arguments.allow_legacy_resume,
                num_workers=arguments.num_workers, device=arguments.device,
            ), validation_path=arguments.validation_data)
        else:
            checkpoint_source = arguments.resume if arguments.resume is not None else arguments.checkpoint
            model = load_checkpoint(checkpoint_source)
            if model.config.vocab_size != tokenizer.vocab_size:
                raise ValueError("checkpoint vocabulary does not match tokenizer")
            if arguments.command == "sft":
                result = run_sft_jsonl(model, tokenizer, arguments.data, arguments.output, SFTOptions(
                    steps=arguments.steps,
                    batch_size=arguments.batch_size,
                    max_length=arguments.max_length,
                    min_context_tokens=arguments.min_context_tokens,
                    target_mode=arguments.target_mode,
                    learning_rate=arguments.learning_rate,
                    min_learning_rate=arguments.min_learning_rate,
                    weight_decay=arguments.weight_decay,
                    adam_beta1=arguments.adam_beta1,
                    adam_beta2=arguments.adam_beta2,
                    adam_eps=arguments.adam_eps,
                    grad_clip=arguments.grad_clip,
                    warmup_steps=arguments.warmup_steps,
                    precision=arguments.precision,
                    gradient_accumulation_steps=arguments.gradient_accumulation,
                    validation_fraction=arguments.validation_fraction,
                    validation_every=arguments.validation_every,
                    validation_batches=arguments.validation_batches,
                    save_every=arguments.save_every,
                    keep_last_checkpoints=arguments.keep_last,
                    generation_every=arguments.generation_every,
                    generation_max_new_tokens=arguments.generation_max_new_tokens,
                    deduplicate_exact=arguments.deduplicate_exact,
                    log_every=arguments.log_every,
                    num_workers=arguments.num_workers,
                    seed=arguments.seed,
                    device=arguments.device,
                    wandb_enabled=arguments.wandb,
                    wandb_project=arguments.wandb_project,
                    wandb_entity=arguments.wandb_entity,
                    wandb_run_name=arguments.wandb_run_name,
                    wandb_run_id=arguments.wandb_run_id,
                    wandb_mode=arguments.wandb_mode,
                    wandb_retry_every_steps=arguments.wandb_retry_every,
                    resume_from=arguments.resume,
                ), validation_path=arguments.validation_data, initial_checkpoint_path=arguments.checkpoint)
            elif arguments.command == "dpo":
                result = run_dpo_jsonl(
                    model,
                    tokenizer,
                    arguments.data,
                    arguments.output,
                    DPOOptions(
                        steps=arguments.steps,
                        batch_size=arguments.batch_size,
                        max_length=arguments.max_length,
                        min_context_tokens=arguments.min_context_tokens,
                        target_mode=arguments.target_mode,
                        learning_rate=arguments.learning_rate,
                        min_learning_rate=arguments.min_learning_rate,
                        weight_decay=arguments.weight_decay,
                        adam_beta1=arguments.adam_beta1,
                        adam_beta2=arguments.adam_beta2,
                        adam_eps=arguments.adam_eps,
                        beta=arguments.beta,
                        grad_clip=arguments.grad_clip,
                        warmup_steps=arguments.warmup_steps,
                        precision=arguments.precision,
                        gradient_accumulation_steps=arguments.gradient_accumulation,
                        validation_fraction=arguments.validation_fraction,
                        validation_every=arguments.validation_every,
                        validation_batches=arguments.validation_batches,
                        save_every=arguments.save_every,
                        keep_last_checkpoints=arguments.keep_last,
                        generation_every=arguments.generation_every,
                        generation_max_new_tokens=arguments.generation_max_new_tokens,
                        deduplicate_exact=arguments.deduplicate_exact,
                        log_every=arguments.log_every,
                        num_workers=arguments.num_workers,
                        seed=arguments.seed,
                        device=arguments.device,
                        wandb_enabled=arguments.wandb,
                        wandb_project=arguments.wandb_project,
                        wandb_entity=arguments.wandb_entity,
                        wandb_run_name=arguments.wandb_run_name,
                        wandb_run_id=arguments.wandb_run_id,
                        wandb_mode=arguments.wandb_mode,
                        wandb_retry_every_steps=arguments.wandb_retry_every,
                        resume_from=arguments.resume,
                    ),
                    validation_path=arguments.validation_data,
                    initial_checkpoint_path=arguments.checkpoint,
                )
            elif arguments.command == "grpo":
                result = run_grpo_jsonl(model, tokenizer, arguments.data, arguments.output, GRPOOptions(
                    steps=arguments.steps, batch_size=arguments.batch_size, group_size=arguments.group_size,
                    max_new_tokens=arguments.max_new_tokens, policy_epochs=arguments.policy_epochs,
                    learning_rate=arguments.learning_rate, min_learning_rate=arguments.min_learning_rate,
                    weight_decay=arguments.weight_decay, adam_beta1=arguments.adam_beta1,
                    adam_beta2=arguments.adam_beta2, adam_eps=arguments.adam_eps,
                    warmup_steps=arguments.warmup_steps, clip_epsilon=arguments.clip_epsilon,
                    beta=arguments.beta, temperature=arguments.temperature,
                    top_k=arguments.top_k or None, grad_clip=arguments.grad_clip,
                    precision=arguments.precision, reference_device=arguments.reference_device,
                    validation_fraction=arguments.validation_fraction,
                    validation_every=arguments.validation_every,
                    validation_prompts=arguments.validation_prompts,
                    save_every=arguments.save_every, keep_last_checkpoints=arguments.keep_last,
                    data_limit=arguments.data_limit, log_every=arguments.log_every,
                    seed=arguments.seed, device=arguments.device,
                    wandb_enabled=arguments.wandb, wandb_project=arguments.wandb_project,
                    wandb_entity=arguments.wandb_entity, wandb_run_name=arguments.wandb_run_name,
                    wandb_run_id=arguments.wandb_run_id, wandb_mode=arguments.wandb_mode,
                    wandb_retry_every_steps=arguments.wandb_retry_every,
                    resume_from=arguments.resume,
                ), validation_path=arguments.validation_data,
                    initial_checkpoint_path=arguments.checkpoint)
            else:
                result = run_agent_grpo_jsonl(model, tokenizer, arguments.data, arguments.output, AgentRLOptions(
                    steps=arguments.steps, batch_size=arguments.batch_size, group_size=arguments.group_size,
                    max_new_tokens=arguments.max_new_tokens, max_turns=arguments.max_turns,
                    policy_epochs=arguments.policy_epochs,
                    learning_rate=arguments.learning_rate, min_learning_rate=arguments.min_learning_rate,
                    weight_decay=arguments.weight_decay, adam_beta1=arguments.adam_beta1,
                    adam_beta2=arguments.adam_beta2, adam_eps=arguments.adam_eps,
                    warmup_steps=arguments.warmup_steps, clip_epsilon=arguments.clip_epsilon,
                    beta=arguments.beta, temperature=arguments.temperature,
                    top_k=arguments.top_k or None, grad_clip=arguments.grad_clip,
                    precision=arguments.precision, reference_device=arguments.reference_device,
                    validation_fraction=arguments.validation_fraction,
                    validation_every=arguments.validation_every,
                    validation_prompts=arguments.validation_prompts,
                    save_every=arguments.save_every, keep_last_checkpoints=arguments.keep_last,
                    data_limit=arguments.data_limit, log_every=arguments.log_every,
                    seed=arguments.seed, device=arguments.device,
                    wandb_enabled=arguments.wandb, wandb_project=arguments.wandb_project,
                    wandb_entity=arguments.wandb_entity, wandb_run_name=arguments.wandb_run_name,
                    wandb_run_id=arguments.wandb_run_id, wandb_mode=arguments.wandb_mode,
                    wandb_retry_every_steps=arguments.wandb_retry_every,
                    resume_from=arguments.resume,
                ), validation_path=arguments.validation_data,
                    initial_checkpoint_path=arguments.checkpoint)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
