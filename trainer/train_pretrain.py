"""Thin entry point for the package pretraining implementation."""

from miniscale.config import MiniScaleConfig
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer
from miniscale.training.common import seed_everything
from miniscale.training.pretrain import SmokePretrainOptions, run_pretrain


def main() -> None:
    seed_everything(42)
    model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
    metrics = run_pretrain(
        model,
        ByteTokenizer(),
        ["MiniScale learns language from next-token prediction. " * 8],
        "artifacts",
        SmokePretrainOptions(),
    )
    print(metrics)


if __name__ == "__main__":
    main()
