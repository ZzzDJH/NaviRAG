import requests
from openai import OpenAI
from typing import Union
import os
from functools import lru_cache

PromptType = Union[str, list[str]]


# ============================
# OpenAI compatible API client
# ============================

@lru_cache(maxsize=1)
def get_api_client() -> OpenAI:
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Please set it before using asking_api()."
        )

    client_kwargs = {
        "api_key": api_key,
    }

    if base_url:
        client_kwargs["base_url"] = base_url.rstrip("/")

    return OpenAI(**client_kwargs)


def asking_api(
    prompt_or_prompts: PromptType,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    top_p: float = 0.9,
    model: str = "gpt-4o",
    max_tokens: int = 4096,
):

    def build_messages(prompt: str) -> list[dict[str, str]]:
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    if isinstance(prompt_or_prompts, str):
        prompts = [prompt_or_prompts]
        is_single_prompt = True
    elif isinstance(prompt_or_prompts, list):
        prompts = prompt_or_prompts
        is_single_prompt = False
    else:
        raise TypeError(
            "prompt_or_prompts must be str or list[str]"
        )

    if not all(isinstance(prompt, str) for prompt in prompts):
        raise TypeError(
            "Every item in prompt_or_prompts must be a string"
        )


    client = get_api_client()

    outputs = []

    for prompt in prompts:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        message = response.choices[0].message

        if message.content is None:
            print(
                "Warning: LLM returned empty content",
                response,
            )
            outputs.append("None")
        else:
            outputs.append(message.content.strip())

    return outputs[0] if is_single_prompt else outputs



# ============================
# ChatLLM wrapper
# ============================


class chatllm:
    """
    Unified LLM interface.

    Supports:
        1. vLLM OpenAI-compatible server
        2. OpenAI-compatible API

    Example:
        llm = chatllm(
            is_vllm=True,
            model_name="llama"
        )

        response = llm.generate(prompt)
    """

    def __init__(
        self,
        model_path: str = None,
        is_vllm: bool = True,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "llama-3"
    ):

        self.is_vllm = is_vllm

        self.model_name = model_name

        self.model_path = model_path

        self.base_url = base_url.rstrip("/")


    def asking_vllm(
        self,
        prompt_or_prompts: PromptType,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 1280,
        enable_thinking: bool = False,
    ):

        prompts = (
            [prompt_or_prompts]
            if isinstance(prompt_or_prompts, str)
            else prompt_or_prompts
        )

        if not isinstance(prompts, list):
            raise TypeError(
                "prompt_or_prompts must be str or list[str]"
            )

        outputs = []

        for prompt in prompts:

            payload = {

                "model": self.model_name,

                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "temperature": temperature,

                "top_p": top_p,

                "max_tokens": max_tokens,

                "seed": 12345,

                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking
                }
            }


            response = requests.post(

                f"{self.base_url}/chat/completions",

                headers={
                    "Content-Type": "application/json"
                },

                json=payload,

                timeout=300
            )


            response.raise_for_status()


            content = (
                response
                .json()
                ["choices"][0]
                ["message"]
                ["content"]
            )


            outputs.append(
                content.strip()
            )


        return (
            outputs[0]
            if isinstance(prompt_or_prompts, str)
            else outputs
        )

    def generate(
        self,
        prompt_or_prompts: PromptType,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 1280,
        model: str = "",
        enable_thinking: bool = False,
    ):

        if self.is_vllm:

            return self.asking_vllm(
                prompt_or_prompts,
                temperature,
                top_p,
                max_tokens,
                enable_thinking
            )

        else:

            return asking_api(
                prompt_or_prompts,
                temperature=temperature,
                top_p=top_p,
                model=model or self.model_name,
                max_tokens=max_tokens,
            )