"""Thin entry point for the package pretraining implementation."""

from miniscale.config import MiniScaleConfig
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer
from miniscale.training.pretrain import PretrainOptions, run_pretrain


def main() -> None:
    model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
    metrics = run_pretrain(
        model,
        ByteTokenizer(),
        ["MiniScale learns language from next-token prediction. " * 8],
        "artifacts",
        PretrainOptions(steps=2, batch_size=2, sequence_length=64),
    )
    print(metrics)


if __name__ == "__main__":
    main()
