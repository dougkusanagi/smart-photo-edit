"""Proteção do app contra entradas excessivas, sem truncar instruções."""
MAX_PROMPT_CHARS = 10_000


class PromptLimitError(ValueError):
    pass


def effective_prompt(prompt: str, prefix: str = '') -> str:
    return ' '.join(part for part in (prefix.strip(), prompt.strip()) if part)


def validate_prompt(prompt: str, prefix: str = '') -> None:
    if len(effective_prompt(prompt, prefix)) > MAX_PROMPT_CHARS:
        raise PromptLimitError('O prompt aceita até 10.000 caracteres, incluindo o texto de apoio das LoRAs. Reduza o texto; nenhuma instrução foi cortada.')
