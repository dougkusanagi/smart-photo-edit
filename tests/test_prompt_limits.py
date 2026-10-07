import pytest
from unittest.mock import AsyncMock

from smart_photo_edit import models
from smart_photo_edit.prompt_limits import MAX_PROMPT_CHARS, validate_prompt
from .test_server import edit_form, make_env, read_events


def test_limit_counts_unicode_characters_and_effective_lora_prompt():
    validate_prompt('😀' * MAX_PROMPT_CHARS)
    validate_prompt('x' * (MAX_PROMPT_CHARS - 4), 'abc')
    with pytest.raises(ValueError, match='10.000'):
        validate_prompt('x' * (MAX_PROMPT_CHARS - 3), 'abc')


async def test_over_limit_is_rejected_before_upload_or_engine_execution(make_env, png, monkeypatch):
    env = await make_env()
    env.state.settings.engine_mode = 'managed'
    monkeypatch.setattr(env.state.runtime, 'ensure', AsyncMock(return_value=True))
    for options in [{'prompt': 'x' * (MAX_PROMPT_CHARS + 1)}, {'prompt': 'x' * MAX_PROMPT_CHARS, 'addons': ['light-blend']}]:
        response = await env.http.post('/api/edit', data=edit_form(png, **options))
        assert response.status in (400, 422)
        assert '10.000' in (await response.json())['error']
    assert not env.comfy.uploads and not env.comfy.prompts
    env.state.runtime.ensure.assert_not_awaited()
    assert not env.state.store.list()['items']
    response = await env.http.post('/api/prompt/enhance', json={'prompt': 'x' * (MAX_PROMPT_CHARS + 1)})
    assert response.status == 422 and not env.state.prompt_ai.busy


async def test_exact_effective_limit_reaches_workflow_and_history_intact(make_env, png, monkeypatch):
    env = await make_env()
    env.state.runtime.url = env.state.settings.comfy_url
    env.state.settings.engine_mode = 'managed'
    monkeypatch.setattr(env.state.runtime, 'ensure', AsyncMock(return_value=True))
    monkeypatch.setattr(env.state.runtime, 'ensure_addons', AsyncMock(return_value=True))
    prefix = models.CATALOG['addons']['light-blend']['prompt']
    prompt = '😀' * (MAX_PROMPT_CHARS - len(prefix) - 1)
    events = await read_events(await env.http.post('/api/edit', data=edit_form(png, prompt=prompt, addons=['light-blend'])))
    item = events[-1]['images'][0]
    assert item['user_prompt'] == prompt
    assert item['prompt'] == prefix + ' ' + prompt and len(item['prompt']) == MAX_PROMPT_CHARS
    assert (await (await env.http.get('/api/status')).json())['limits']['prompt_chars'] == MAX_PROMPT_CHARS
