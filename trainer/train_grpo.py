"""Small standalone single-turn GRPO/RLVR example."""

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.grpo import GRPOOptions, RLTask, run_grpo


def main() -> None:
    model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
    tasks = [RLTask("Return only the result: 2+3", "5"), RLTask("Return only the result: 3*4", "12")]
    metrics = run_grpo(
        model,
        ByteTokenizer(),
        tasks,
        "artifacts",
        GRPOOptions(steps=1, group_size=2, max_new_tokens=8),
    )
    print(metrics)


if __name__ == "__main__":
    main()
