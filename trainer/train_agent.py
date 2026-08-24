"""Small standalone multi-turn tool-use Agent GRPO example."""

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.agent_env import CalculatorTask
from miniscale.training.agent_rl import AgentRLOptions, run_agent_grpo


def main() -> None:
    tasks = [CalculatorTask("Use the calculator for 2+3, then answer.", "2+3", "5")]
    metrics = run_agent_grpo(
        MiniScaleForCausalLM(MiniScaleConfig.smoke()),
        ByteTokenizer(),
        tasks,
        "artifacts",
        AgentRLOptions(steps=1, group_size=2, max_turns=2, max_new_tokens=12),
    )
    print(metrics)


if __name__ == "__main__":
    main()
