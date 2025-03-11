from vllm import LLM, SamplingParams


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

if __name__ == "__main__":

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="float32",
        device="cpu",
        trust_remote_code=True,
        enforce_eager=True,
        disable_async_output_proc=True,
        max_model_len=2048
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
