"""Thin entry point for the package pretraining implementation."""

from miniscale.config import MiniScaleConfig
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer
from miniscale.training.configs.pretrain import SmokePretrainOptions
from miniscale.training.core.runtime import seed_everything
from miniscale.training.stages.pretrain import run_pretrain


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
