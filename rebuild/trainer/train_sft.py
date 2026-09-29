"""Small standalone SFT example."""

from miniscale.training.core.runtime import seed_everything

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.sft.config import SmokeSFTOptions
from miniscale.training.sft import run_sft


def main() -> None:
    seed_everything(42)
    conversations = [
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2+3?"},
            {"role": "assistant", "content": "5"},
        ],
        [
            {"role": "user", "content": "Use the calculator for 4*6."},
            {
                "role": "assistant",
                "content": '<tool_call>{"name":"calculator","arguments":{"expression":"4*6"}}</tool_call>',
            },
        ],
    ]
    metrics = run_sft(
        MiniScaleForCausalLM(MiniScaleConfig.smoke()),
        ByteTokenizer(),
        conversations,
        "artifacts",
        SmokeSFTOptions(steps=2, batch_size=2),
    )
    print(metrics)


if __name__ == "__main__":
    main()
