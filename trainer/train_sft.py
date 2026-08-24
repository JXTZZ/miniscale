"""Small standalone SFT example."""

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.sft import SFTOptions, run_sft


def main() -> None:
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
        SFTOptions(steps=2, batch_size=2),
    )
    print(metrics)


if __name__ == "__main__":
    main()
