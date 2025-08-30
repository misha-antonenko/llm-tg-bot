from __future__ import annotations

import asyncio as aio
import copy
import html
import io
import json
import logging
import os
import textwrap
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterable, Callable, Iterable
from functools import cached_property, wraps
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

# was introduced with `litellm`
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic', append=True)

import html2text
import httpx
import litellm
import msgpack
import openai
import pydantic
import readability
import tap
import telegram
import telegram.ext
import telegramify_markdown
import wolframalpha
import yaml
from googleapiclient.discovery import build
from litellm.cost_calculator import completion_cost
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackContext,
    CommandHandler,
    ContextTypes,
    Defaults,
    ExtBot,
    MessageHandler,
)

import _pdf
from _markdown import split_markdown_into_chunks
from _memory import Memory
from _persistence import Persistence
from _retries import retry
from _state import State

if TYPE_CHECKING:
    from litellm.files.main import ModelResponse
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


class ProviderConfig(State):
    api_key: str


class OpenaiConfig(ProviderConfig):
    pass


class AnthropicConfig(ProviderConfig):
    pass


class GaiConfig(ProviderConfig):
    pass


class XaiConfig(ProviderConfig):
    pass


class GoogleSearchConfig(State):
    api_key: str
    engine_id: str


class MathpixConfig(State):
    api_id: str
    api_key: str


class Config(State):
    model_config: ClassVar[pydantic.ConfigDict] = dict(**State.model_config, frozen=True)  # pyright: ignore[reportAssignmentType, reportCallIssue]

    openai_config: OpenaiConfig
    anthropic_config: AnthropicConfig
    gai_config: GaiConfig
    xai_config: XaiConfig
    telegram_bot_token: str
    wolfram_app_id: str
    google_search_config: GoogleSearchConfig
    owner: int
    owner_nickname: str
    log_path: str = 'bot.log'
    state_dir: str = 'state'
    allowed_chat_ids: set[int] | None = None
    mathpix_config: MathpixConfig
    pdf_cals_base_dir: str = 'pdf_cals'

    @property
    def state_path(self):
        return os.path.join(self.state_dir, 'state.pickle')


config: Config


async def query_wolframalpha(query: str) -> str:
    """
    Queries WolframAlpha. Returns a list of results in plaintext.
    """
    client = wolframalpha.Client(config.wolfram_app_id)
    res = await client.aquery(query)
    buf = io.StringIO()
    for pod_idx, pod in enumerate(res.pods, start=1):
        print(f'{pod_idx}. {pod.title}', file=buf)
        for subpod_idx, subpod in enumerate(pod.subpods, start=1):
            if (text := subpod.plaintext) is None:
                continue
            text = textwrap.indent(text, prefix='    ')
            print(
                f'  {pod_idx}.{subpod_idx}.\n{text}',
                file=buf,
            )
        print('\n', file=buf)
    res = buf.getvalue()
    if not res:
        logging.warning('Got an empty response from WolframAlpha')
    else:
        logging.debug('Got a response from WolframAlpha:\n%s', res)
    return res


async def search_google(query: str, page_idx: int = 0) -> list[dict[str, str]]:
    """
    Searches Google for the given query `query`. Returns a list of 10 web page titles together with their
    URLs and short snippets (typically under 300 characters in length).

    `page_idx` is the pagination index (so that passing 0 will return the first 10 results
    that Google knows, passing 1 will return the next 10, etc.). `page_idx` must be in the range
    [0, 9].

    Some operators the search supports:
    1. `site:example.com` - restricts the search to the given site
    2. `filetype:pdf` - restricts the search to the given file type
    3. `intitle:word` - restricts the search to pages with the given word in the title
    4. `inurl:word` - restricts the search to pages with the given word in the URL
    5. `"s0m3 str1ng"` - restricts the search to pages that contain `s0m3 str1ng` literally
    """
    # TODO(atm): these clients are not thread-safe, so we create a new one each time
    client = build(
        'customsearch',
        'v1',
        developerKey=config.google_search_config.api_key,
        num_retries=9,
    ).cse()
    n_results_per_page = 10
    response = client.list(
        q=query,
        cx=config.google_search_config.engine_id,
        num=n_results_per_page,
        start=1 + n_results_per_page * page_idx,  # the indexing is 1-based
    ).execute()
    try:
        n_results = int(response['searchInformation']['totalResults'])
        if n_results == 0:
            return []
        return [
            dict(
                title=item['title'],
                url=item['link'],
                snippet=item['snippet'],
            )
            for item in response['items']
        ]
    except KeyError as exc:
        raise RuntimeError(f'Failed to parse the response from Google: {response!r}') from exc


def _parse_with_readability(text: str) -> str:
    doc = readability.Document(text)

    # Extract the main content
    content = doc.summary()

    # Convert HTML to Markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_tables = False
    markdown_content = h.handle(content)

    return markdown_content


_pdf_cals: _pdf.Cals


async def fetch_main_text(url: str) -> str:
    """
    Fetches the content of the page at the given URL and then:
    1. If the content is a PDF file, uses Mathpix to extract its content into Markdown and returns
      the latter.
    2. If the content is a UTF-8 HTML page, extracts the "main" text from it using the `readability`
      library, converts this HTML text into Markdown, and returns the latter.
    3. Otherwise the behavior is undefined.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if _pdf.is_pdf(response.content):
        return await _pdf_cals.load_pdf(_pdf.Attrs(name=url), response.content)

    return _parse_with_readability(response.text)


_PYTHON_TYPE_TO_JSON_SCHEMA_TYPE = {
    'str': 'string',
    'int': 'integer',
    'float': 'number',
    'bool': 'boolean',
    'list': 'array',
    'dict': 'object',
}


def _get_json_schema_type(t: type | str) -> str:
    return _PYTHON_TYPE_TO_JSON_SCHEMA_TYPE[t.__name__ if isinstance(t, type) else t]


Tools = dict[str, Callable[..., Any]]

_BASE_TOOLS: Tools = {
    f.__name__: f
    for f in map(
        retry(OSError),
        [
            search_google,
            fetch_main_text,
            query_wolframalpha,
        ],
    )
}
"""
A dictionary of tools that can be called by the model.

When adding new ones, make sure that the annotations used are present in
`_PYTHON_TYPE_TO_JSON_SCHEMA_TYPE`.
"""


def _validate_tools_for_openai(tools: Tools):
    for tool in tools.values():
        name = tool.__name__
        doc = tool.__doc__
        assert doc is not None
        assert len(doc) <= 1024, (
            f'The description of the tool {name!r} is {len(doc) - 1024} characters too long'
        )
        assert len(name) <= 64, f'The name of the tool {name!r} is too long'


class Args(tap.Tap):
    config_path: str = 'config.yaml'


class UserData(State):
    user_id: int | None = None
    user_name: str | None = None
    usage: deque[tuple[int, float]] = pydantic.Field(default_factory=deque)

    @pydantic.field_validator('usage', mode='before')
    @classmethod
    def _validate_usage(cls, v: Iterable) -> deque[tuple[int, float]]:
        return deque(tuple(e) for e in v)

    @pydantic.field_serializer('usage')
    def _serialize_usage(self, v: deque[tuple[int, float]]) -> list[tuple[int, float]]:
        return list(v)

    @property
    def total_cost_dollars(self) -> float:
        return sum(cost for _, cost in self.usage)

    @property
    def name_with_id(self) -> str:
        return f'{self.user_name} ({self.user_id})'

    def register_usage(self, now_ns: int, cost_dollars: float):
        self.usage.append((now_ns, cost_dollars))


def _make_memory(chat_id: int) -> Memory:
    return Memory(Path(config.state_dir, f'chat_{chat_id}', 'memos'))


def _escape(x: Any) -> str:
    return telegramify_markdown.escape_markdown(str(x))


def _migrate_provider_id(data: dict[str, Any], field_name: str):
    provider_id = data.get(field_name)
    assert isinstance(provider_id, str | None)
    if provider_id == 'openai-o3-mini-high':
        data[field_name] = 'openai-thinking'


class MessageToSend(pydantic.BaseModel):
    text: str
    do_notify: bool = True


class LmChat(ABC, State):
    lm_provider: str
    chat_id: int
    memory: Memory = pydantic.Field(exclude=True)
    tools: Tools = pydantic.Field(exclude=True)
    history: list[dict[str, Any]] = pydantic.Field(default_factory=list)
    prompt_update_history: list[PromptUpdate] = pydantic.Field(default_factory=list)

    @pydantic.model_validator(mode='before')
    @classmethod
    def _validate_before(
        cls, data: dict[str, Any], info: pydantic.ValidationInfo
    ) -> dict[str, Any]:
        chat_id = data.get('chat_id')
        if chat_id is None:
            if info.context is None:
                raise pydantic.ValidationError('no `chat_id` in the data or context')
            chat_id = data['chat_id'] = info.context['chat_id']
        data['memory'] = memory = _make_memory(chat_id)
        data['tools'] = cls._make_tools(memory)
        _migrate_provider_id(data, 'lm_provider')
        return data

    @classmethod
    def _make_tools(cls, memory: Memory) -> Tools:
        tools = _BASE_TOOLS.copy()

        async def set_memo(path: str, value: str = ''):
            """
            Sets the contents of the file at the given path. `path` must be a relative Unix path
            without whitespace in segments, e.g. `cooking/recipes/pancakes`. If `value` is empty
            or not specified, instead deletes the file at the given path.

            Memos are your long-term memory. They allow information to escape your context
            window.
            """
            memory.set_memo(path, value)

        async def append_to_memo(path: str, value: str):
            """
            Appends the given value to the file at the given path. If the file does not exist, it
            will be created. See the doc for `set_memo` for more information.
            """
            memory.append_to_memo(path, value)

        async def get_memo(pattern: str) -> str | None:
            """
            Retrieves all memos with paths that match the given pattern, concatenates their
            contents surrounded by `<memo>` tags, and returns the resulting string.
            """
            paths = memory.glob_memos(pattern)
            paths.sort()
            res = io.StringIO()
            for path in paths:
                print(f'<memo path={path!r}>{memory.get_memo(path)}</memo>', file=res, end='')
            return res.getvalue()

        async def glob_memos(pattern: str = '**') -> list[str]:
            """
            Returns a list of paths to files (not directories) that match the given pattern. `*` in
            `pattern` matches any non-negative number of characters that are not equal to `/`.
            `**` is similar, but can also include `/`. The default value for `pattern` is `**`.
            """
            res = memory.glob_memos(pattern)
            logging.info('Globbed %d memos with pattern %r: %r', len(res), pattern, res)
            return res

        for tool in (
            set_memo,
            append_to_memo,
            get_memo,
            glob_memos,
        ):
            tools[tool.__name__] = tool

        return tools

    def _update_system_prompt(self, context: Context):
        assert (chat_data := context.chat_data) is not None
        if chat_data.system_prompt != self.system_prompt:
            self.prompt_update_history.append(
                PromptUpdate(
                    prompt=chat_data.system_prompt,
                    turn_idx=len(self.history),
                ),
            )
            self.system_prompt = chat_data.system_prompt

    @abstractmethod
    async def send_message(
        self,
        message: str,
        context: Context,
    ) -> AsyncIterable[MessageToSend]:
        pass


class PromptUpdate(pydantic.BaseModel):
    prompt: str
    turn_idx: int


class LlmResponseFormatError(Exception):
    pass


class UnexpectedResponseTypeError(TypeError, LlmResponseFormatError):
    pass


class BadCompletionChoices(LlmResponseFormatError):
    pass


class OpenaiChat(LmChat):
    lm_provider: Literal['openai'] = 'openai'  # pyright: ignore[reportIncompatibleVariableOverride]
    lm_id: str = 'gpt-5'
    responses_prev_id: str | None = None
    ADDITIONAL_REQUEST_PARAMS: ClassVar[dict[str, Any]] = dict(
        reasoning={'effort': 'minimal'},
    )

    @classmethod
    def _assign_cache_breakpoints_to_tools(cls, tools: list[dict[str, Any]]):
        # No-op for Responses API: tools schema does not support cache_control
        pass

    @classmethod
    def _assign_cache_breakpoints_to_messages(cls, new_history: list):
        for turn in reversed(new_history):
            content = cls._prepare_content(turn)
            if not content:
                continue
            last_block: dict[str, Any] = content[-1]
            assert isinstance(last_block, dict), type(last_block)
            if 'cache_control' in last_block:
                break
        n_breakpoints_seen = 0
        did_cache_system_prompt = False
        for turn in reversed(new_history):
            content = cls._prepare_content(turn)
            if not content:
                continue
            last_block: dict[str, Any] = content[-1]
            assert isinstance(last_block, dict), type(last_block)
            if turn['role'] == 'system':
                # this is a system message, cache the last of them
                if did_cache_system_prompt:
                    continue
                last_block['cache_control'] = dict(type='ephemeral')
                did_cache_system_prompt = True
            elif n_breakpoints_seen == 0:
                # add a breakpoint to the last message in the new history
                last_block['cache_control'] = dict(type='ephemeral')
                n_breakpoints_seen += 1
            elif n_breakpoints_seen == 1:
                # count the second one from the end
                n_breakpoints_seen += 'cache_control' in last_block
            elif n_breakpoints_seen == 2:
                # remove all other breakpoints (if they were saved by a previous version of the code)
                last_block.pop('cache_control', None)

    @cached_property
    def _prepared_tools(self):
        # Build OpenAI Responses API ToolParam (FunctionToolParam) items
        res: list[dict[str, Any]] = []
        for fn in self.tools.values():
            param_schema = dict(
                type='object',
                properties={
                    arg_name: dict(type=_get_json_schema_type(arg_type))
                    for arg_name, arg_type in fn.__annotations__.items()
                    if arg_name != 'return'
                },
                additionalProperties=False,
                required=[
                    arg_name
                    for arg_name, _arg_type in fn.__annotations__.items()
                    if arg_name != 'return'
                ],
            )
            res.append(
                dict(
                    type='function',
                    name=fn.__name__,
                    description=fn.__doc__,
                    parameters=param_schema,
                    strict=True,
                )
            )
        self._assign_cache_breakpoints_to_tools(res)
        return res

    async def _evaluate_function_call(self, name: str, args: str) -> str:
        try:
            result = await self.tools[name](**json.loads(args))
        except BaseException as exc:
            response = dict(exception=repr(exc))
            logging.warning('Tool %r raised an exception:', name, exc_info=exc)
        else:
            response = dict(result=result)
        logging.debug('Function %r returned %r', name, response)
        return repr(response)

    @property
    def _api_key(self) -> str:
        return config.openai_config.api_key

    @property
    def _litellm_provider(self) -> str:
        return self.lm_provider

    @retry(
        openai.APIError,
        OSError,
        LlmResponseFormatError,
        excluded_exc_classes=(openai.BadRequestError,),
    )
    async def _complete_message(
        self, new_history: list[ChatCompletionMessageParam]
    ) -> litellm.ResponsesAPIResponse:
        model = os.path.join(self._litellm_provider, self.lm_id)
        logging.info('Completing with %r', model)
        completion = await litellm.aresponses(
            input=cast('Any', new_history),
            model=model,
            tools=cast('Any', self._prepared_tools),
            previous_response_id=self.responses_prev_id,
            api_key=self._api_key,
            **self.ADDITIONAL_REQUEST_PARAMS,
        )
        if not isinstance(completion, litellm.ResponsesAPIResponse):
            raise UnexpectedResponseTypeError(type(completion))
        self.responses_prev_id = completion.id
        return completion

    def _register_usage(self, usage: ModelResponse | None, context: Context):
        if usage is None:
            return
        user_data = context.user_data
        if user_data is None:
            return
        now = time.time_ns()
        user_data.register_usage(now, completion_cost(usage))

    @classmethod
    def _prepare_content(cls, turn: ChatCompletionMessageParam) -> list[dict[str, Any]]:
        content = turn['content']  # pyright: ignore[reportTypedDictNotRequiredAccess]
        if isinstance(content, str):
            # could be saved by a previous version of the code
            turn['content'] = content = [dict(text=content, type='text')]  # pyright: ignore[reportGeneralTypeIssues]
        if content is None:
            # tool calls might have this
            turn['content'] = content = []  # pyright: ignore[reportGeneralTypeIssues]
        assert isinstance(content, list), type(content)
        return content  # pyright: ignore[reportReturnType]

    def _build_responses_input(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert Chat Completions-style history into OpenAI Responses API input list.

        - user/system/developer messages with text -> message items (content as plain string)
        - tool role -> function_call_output items
        - skip assistant messages (tracked via previous_response_id)
        """
        items: list[dict[str, Any]] = []
        for turn in history:
            role = turn.get('role')
            content = turn.get('content')

            if role == 'tool':
                call_id = turn.get('tool_call_id')
                output = turn.get('content', '')
                if call_id:
                    items.append(
                        dict(
                            type='function_call_output',
                            call_id=str(call_id),
                            output=str(output),
                        )
                    )
                continue

            if role in ('user', 'system', 'developer'):
                text = ''
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            t = block.get('text')
                            if t:
                                parts.append(str(t))
                    text = '\n'.join(parts)

                # Only include messages that actually have textual content
                if text:
                    items.append(
                        dict(
                            type='message',
                            role=role,
                            content=text,
                        )
                    )
        return items

    def _build_responses_delta(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Build only the trailing delta items since the last assistant turn.
        - Include contiguous tail of history until an 'assistant' role is encountered.
        - Map using _build_responses_input to proper Responses API items.
        """
        tail: list[dict[str, Any]] = []
        for turn in reversed(history):
            role = turn.get('role')
            if role == 'assistant':
                break
            tail.append(turn)
        tail.reverse()
        return self._build_responses_input(tail)

    @property
    def system_prompt(self) -> str:
        if not self.history or self.history[0]['role'] != 'system':
            return ''
        blocks = self._prepare_content(self.history[0])  # pyright: ignore[reportArgumentType]
        assert isinstance(blocks, list), type(blocks)
        assert len(blocks) == 1, len(blocks)
        return blocks[0]['text']

    @system_prompt.setter
    def system_prompt(self, value: str):
        if self.history and self.history[0]['role'] == 'system':
            self.history[0]['content'] = value
        else:
            self.history.insert(0, dict(role='system', content=value))

    async def send_message(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        message: str,
        context: Context,
    ) -> AsyncIterable[MessageToSend]:
        self._update_system_prompt(context)
        new_history: list[ChatCompletionMessageParam] = [
            *self.history,
            dict(role='user', content=message),
        ]  # pyright: ignore[reportAssignmentType]
        while True:
            responses_input = (
                self._build_responses_delta(cast('list[dict[str, Any]]', new_history))
                if self.responses_prev_id
                else self._build_responses_input(cast('list[dict[str, Any]]', new_history))
            )
            completion = await self._complete_message(cast('Any', responses_input))
            self._register_usage(completion, context)  # pyright: ignore[reportArgumentType]

            # Extract text and tool calls from Responses API output
            resp_output = completion.output or []
            resp_text_parts: list[str] = []
            tool_calls = []
            for item in resp_output:
                # Message items -> accumulate output_text parts
                if isinstance(item, dict):
                    itype = item.get('type')
                    if itype == 'message':
                        content_list = item.get('content', []) or []
                        for part in content_list:
                            if isinstance(part, dict):
                                if part.get('type') in ('output_text', 'text'):
                                    text_val = part.get('text')
                                    if text_val:
                                        resp_text_parts.append(str(text_val))
                        continue
                    elif itype == 'function_call':
                        name = item.get('name', '')
                        args = item.get('arguments', '{}')
                        call_id = (
                            item.get('call_id') or item.get('id') or f'call_{len(tool_calls) + 1}'
                        )
                        tool_calls.append(
                            dict(
                                index=len(tool_calls),
                                id=call_id,
                                type='function',
                                function=dict(name=name or '', arguments=args or '{}'),
                            )
                        )
                        continue
                    else:
                        raise ValueError(f'unexpected item type {itype!r}; whole item: {item}')
                # Typed SDK objects
                elif isinstance(item, ResponseOutputMessage):
                    for part in item.content or []:
                        if isinstance(part, ResponseOutputText):
                            if part.text:
                                resp_text_parts.append(str(part.text))
                        elif isinstance(part, ResponseOutputRefusal):
                            # ignore refusal text for normal content aggregation
                            pass
                    continue
                elif isinstance(item, ResponseFunctionToolCall):
                    name = item.name
                    args = item.arguments
                    call_id = item.call_id or (item.id or f'call_{len(tool_calls) + 1}')
                    tool_calls.append(
                        dict(
                            index=len(tool_calls),
                            id=call_id,
                            type='function',
                            function=dict(name=name or '', arguments=args or '{}'),
                        )
                    )
                    continue
                elif isinstance(item, ResponseReasoningItem):
                    # don't display thoughts to the user
                    continue
                else:
                    raise TypeError(f'unexpected output type {type(item)}')

            content_text = ''.join(resp_text_parts).strip()

            # Append assistant turn with content and tool calls (Chat Completions style for compatibility)
            assistant_turn: dict[str, Any] = dict(role='assistant')
            if content_text:
                assistant_turn['content'] = content_text
            if tool_calls:
                assistant_turn['tool_calls'] = tool_calls
            new_history.append(cast('Any', assistant_turn))  # pyright: ignore[reportArgumentType]
            self.history = new_history  # pyright: ignore[reportAttributeAccessIssue]

            if content_text:
                yield MessageToSend(text=content_text)

            call_futures = (
                [
                    aio.create_task(
                        self._evaluate_function_call(
                            tc['function']['name'] or '',
                            tc['function']['arguments'] or '{}',
                        )
                    )
                    for tc in tool_calls
                ]
                if tool_calls
                else None
            )

            if call_futures:
                tool_call_reports: list[str] = []
                for tc in tool_calls:
                    try:
                        args_dict = json.loads(tc['function']['arguments'] or '{}')
                        args = ', '.join(f'{_escape(k)}=`{v}`' for k, v in args_dict.items())
                    except json.JSONDecodeError:
                        args = '<failed to parse the arguments>'
                    tool_call_reports.append(f'{tc["function"]["name"]}({args})')
                yield MessageToSend(
                    text=f'_Executing {", ".join(tool_call_reports)}..._', do_notify=False
                )

                call_results = await aio.gather(*call_futures)
                for tc, call_result in zip(tool_calls, call_results, strict=True):
                    new_history.append(
                        cast(
                            'Any',
                            dict(
                                role='tool',
                                tool_call_id=tc['id'],
                                content=call_result,
                            ),
                        )
                    )  # pyright: ignore[reportArgumentType]
                # Continue loop to send tool results back
                continue

            # No tool calls -> we're done
            break


class AnthropicChat(OpenaiChat):
    lm_provider: Literal['anthropic'] = 'anthropic'  # pyright: ignore[reportIncompatibleVariableOverride]
    lm_id: str = 'claude-sonnet-4-20250514'
    ADDITIONAL_REQUEST_PARAMS: ClassVar[dict[str, Any]] = {
        **OpenaiChat.ADDITIONAL_REQUEST_PARAMS,
        'headers': {
            'anthropic-beta': 'token-efficient-tools-2025-02-19',
        },
    }

    @property
    def _litellm_provider(self):
        return 'anthropic'

    @property
    def _api_key(self) -> str:
        return config.anthropic_config.api_key

    @classmethod
    def _check_roles(cls, history: list[dict[str, Any]]):
        current_role = None
        for turn in history:
            new_role = turn['role']
            if current_role is None:
                allowed_roles = ('user', 'system')
            elif current_role == 'system':
                allowed_roles = ('user',)
            elif current_role == 'assistant':
                allowed_roles = ('user', 'tool')
            elif current_role == 'user':
                allowed_roles = ('assistant',)
            elif current_role == 'tool':
                allowed_roles = ('user', 'assistant', 'tool')
            else:
                raise ValueError(f'Unexpected role: {current_role!r}')
            if new_role not in allowed_roles:
                raise ValueError(f'Role {new_role!r} cannot follow {current_role!r}')
            current_role = new_role

    @classmethod
    def _migrate_from_anthropic_format(cls, old_history: list[dict[str, Any]]):
        """
        migrate from the Anthropic-specific history format:
         ```json
         {
            "role": "assistant",
            "content": [
              {
                "text": "<thinking>To answer this question, I should check if there's any stored information about the user.</thinking>",
                "type": "text"
              },
              {
                "id": "toolu_01HFbD2842MD8TBym1F7cMCU",
                "input": {
                  "path": "/"
                },
                "name": "list_directory",
                "type": "tool_use"
              }
            ]
          },
          {
            "role": "user",
            "content": [
              {
                "tool_use_id": "toolu_01HFbD2842MD8TBym1F7cMCU",
                "type": "tool_result",
                "content": "\"PermissionError(13, 'Permission denied')\"",
                "is_error": true
              }
            ]
          },
          ```
        to the OpenAI-specific history format:
          ```json
          {
            "content": "<thinking>To answer this question, I should check if there's any stored information about the user.</thinking>",
            "role": "assistant",
            "tool_calls": [
              {
                "index": 1,
                "function": {
                  "arguments": "{\"path\": \"/\"}",
                  "name": "list_directory"
                },
                "id": "toolu_01HFbD2842MD8TBym1F7cMCU",
                "type": "function"
              }
            ],
            "function_call": null
          },
          {
            "role": "tool",
            "tool_call_id": "toolu_01HFbD2842MD8TBym1F7cMCU",
            "name": "list_directory",
            "content": "\"PermissionError(13, 'Permission denied')\""
          }
          ```
        """
        old_history = copy.deepcopy(old_history)
        new_history: list[dict[str, Any]] = []
        tool_name_by_call_id: dict[str, str] = {}
        for old_turn in old_history:
            role = old_turn['role']
            content = cls._prepare_content(old_turn)  # pyright: ignore[reportArgumentType]
            assert isinstance(content, list), type(content)
            if role == 'assistant':
                tool_calls = []
                for block in content:
                    assert isinstance(block, dict), type(block)
                    block_type = block['type']
                    if block_type == 'tool_use':
                        tool_name_by_call_id[block['id']] = block['name']
                        tool_calls.append(
                            dict(
                                index=len(tool_calls) + 1,
                                function=dict(
                                    arguments=json.dumps(block['input']),
                                    name=block['name'],
                                ),
                                id=block['id'],
                                type='function',
                            )
                        )
                if tool_calls:
                    content = old_turn['content'] = [
                        block for block in content if block['type'] != 'tool_use'
                    ]
                    old_turn['tool_calls'] = tool_calls
                    old_turn['function_call'] = None
                new_history.append(old_turn)
            elif role == 'user':
                tool_results = []
                for block in content:
                    assert isinstance(block, dict), type(block)
                    block_type = block['type']
                    if block_type == 'tool_result':
                        tool_results.append(
                            dict(
                                tool_call_id=block['tool_use_id'],
                                content=block['content'],
                            )
                        )
                if tool_results:
                    for tool_result in tool_results:
                        new_turn = dict(
                            role='tool',
                            tool_call_id=tool_result['tool_call_id'],
                            content=tool_result['content'],
                            name=tool_name_by_call_id[tool_result['tool_call_id']],
                        )
                        new_history.append(new_turn)
                    content = old_turn['content'] = [
                        block for block in content if block['type'] != 'tool_result'
                    ]
                    if content:
                        new_history.append(old_turn)
                else:
                    new_history.append(old_turn)
            else:
                new_history.append(old_turn)
        assert not new_history or new_history[0]['role'] in ('system', 'user')
        cls._check_roles(new_history)
        return new_history

    @pydantic.field_validator('history', mode='after')
    @classmethod
    def _validate_history(cls, old_history: list[dict[str, Any]]):
        try:
            return cls._migrate_from_anthropic_format(old_history)
        except Exception as exc:
            raise ValueError(f'Failed to validate:\n{traceback.format_exc()}') from exc


class AnthropicThinkingChat(AnthropicChat):
    lm_provider: Literal['anthropic-thinking'] = 'anthropic-thinking'  # pyright: ignore[reportIncompatibleVariableOverride]
    ADDITIONAL_REQUEST_PARAMS: ClassVar[dict[str, Any]] = {
        **AnthropicChat.ADDITIONAL_REQUEST_PARAMS,
        'max_tokens': 64000,
        'thinking': {
            'type': 'enabled',
            'budget_tokens': 32000,
        },
    }


class GaiChat(OpenaiChat):
    lm_provider: Literal['google'] = 'google'  # pyright: ignore[reportIncompatibleVariableOverride]
    lm_id: str = 'gemini-2.5-flash'

    @classmethod
    def _assign_cache_breakpoints_to_tools(cls, tools: list[dict[str, Any]]):
        pass

    @classmethod
    def _assign_cache_breakpoints_to_messages(cls, new_history: list):
        pass

    @property
    def _api_key(self) -> str:
        return config.gai_config.api_key

    @property
    def _litellm_provider(self) -> str:
        return 'gemini'


class GaiThinkingChat(GaiChat):
    lm_provider: Literal['google-thinking'] = 'google-thinking'  # pyright: ignore[reportIncompatibleVariableOverride]
    lm_id: str = 'gemini-2.5-pro'
    ADDITIONAL_REQUEST_PARAMS: ClassVar[dict[str, Any]] = dict(
        reasoning={'effort': 'high'},
    )


class OpenaiThinkingChat(OpenaiChat):
    lm_provider: Literal['openai-thinking'] = 'openai-thinking'  # pyright: ignore[reportIncompatibleVariableOverride]
    lm_id: str = 'gpt-5'
    ADDITIONAL_REQUEST_PARAMS: ClassVar[dict[str, Any]] = dict(
        reasoning={'effort': 'high'},
    )

    @property
    def _litellm_provider(self) -> str:
        return 'openai'


class XaiChat(OpenaiChat):
    lm_provider: Literal['xai'] = 'xai'  # pyright: ignore[reportIncompatibleVariableOverride]
    lm_id: str = 'grok-4'

    @classmethod
    def _assign_cache_breakpoints_to_tools(cls, tools: list[dict[str, Any]]):
        pass

    @classmethod
    def _assign_cache_breakpoints_to_messages(cls, new_history: list):
        pass

    @property
    def _api_key(self) -> str:
        return config.xai_config.api_key

    @property
    def _litellm_provider(self) -> str:
        return 'xai'


ProviderId = Literal[
    'anthropic-thinking',
    'anthropic',
    'google-thinking',
    'google',
    'openai-thinking',
    'openai',
    'xai',
]
AnyChat = (
    AnthropicThinkingChat
    | AnthropicChat
    | GaiThinkingChat
    | GaiChat
    | OpenaiThinkingChat
    | OpenaiChat
    | XaiChat
)
_CHAT_CLS_BY_PROVIDER_ID: dict[str, type[LmChat]] = dict(
    zip(ProviderId.__args__, AnyChat.__args__, strict=True)
)
for provider_id, chat_cls in _CHAT_CLS_BY_PROVIDER_ID.items():
    ann = chat_cls.model_fields['lm_provider'].annotation
    args = getattr(ann, '__args__', None)
    if args:
        try:
            assert provider_id == args[0]
        except Exception as exc:
            raise AssertionError(f'{provider_id!r} does not match with {chat_cls!r}') from exc


class Conversation(State):
    id: int
    lm_chat: AnyChat = pydantic.Field(discriminator='lm_provider')

    @pydantic.model_validator(mode='before')
    def _validate_before(cls, data: dict[str, Any]) -> dict[str, Any]:
        if (lm_chat := data.get('lm_chat')) and isinstance(lm_chat, dict):
            _migrate_provider_id(lm_chat, 'lm_provider')
        return data

    async def send_message(
        self,
        message: str,
        context: Context,
    ) -> AsyncIterable[MessageToSend]:
        return self.lm_chat.send_message(
            message,
            context,
        )


class ChatData(State):
    chat_id: int | None = None
    conversation: Conversation | None = None
    min_unused_conversation_id: int = 1
    conversation_id_by_message_id: dict[int, int] = pydantic.Field(default_factory=dict)
    system_prompt: str = ''
    provider_id: ProviderId = 'openai'
    autoreset_grace_seconds: int = int(4.5 * 60)
    impersonated_user_id: int | None = None

    @pydantic.model_validator(mode='before')
    def _validate_before(
        cls, data: dict[str, Any], info: pydantic.ValidationInfo
    ) -> dict[str, Any]:
        if 'chat_id' not in data and (context := info.context) is not None:
            data['chat_id'] = context['chat_id']

        _migrate_provider_id(data, 'provider_id')

        return data

    @classmethod
    def _make_conversation_file_name(cls, conversation_id: int) -> str:
        return f'conversation_{conversation_id}.msgpack'

    def _make_chat(self) -> AnyChat:
        chat_id = self.chat_id
        if chat_id is None:
            raise ValueError('`chat_id` is not set')
        if self.provider_id.startswith('openai'):
            if self.provider_id == 'openai':
                chat_cls = OpenaiChat
            elif self.provider_id == 'openai-thinking':
                chat_cls = OpenaiThinkingChat
            else:
                chat_cls = None
            if chat_cls:
                return chat_cls.model_validate(
                    dict(
                        chat_id=chat_id,
                        history=(
                            [dict(role='system', content=self.system_prompt)]
                            if self.system_prompt
                            else []
                        ),
                    )
                )
        elif self.provider_id.startswith('anthropic'):
            if self.provider_id == 'anthropic':
                chat_cls = AnthropicChat
            elif self.provider_id == 'anthropic-thinking':
                chat_cls = AnthropicThinkingChat
            else:
                chat_cls = None
            if chat_cls:
                return chat_cls.model_validate(
                    dict(
                        chat_id=chat_id,
                        system_prompt=self.system_prompt,
                    )
                )
        elif self.provider_id.startswith('google'):
            if self.provider_id == 'google':
                chat_cls = GaiChat
            elif self.provider_id == 'google-thinking':
                chat_cls = GaiThinkingChat
            else:
                chat_cls = None
            if chat_cls:
                return chat_cls.model_validate(
                    dict(
                        chat_id=chat_id,
                        history=(
                            [dict(role='system', content=self.system_prompt)]
                            if self.system_prompt
                            else []
                        ),
                    )
                )
        elif self.provider_id.startswith('xai'):
            if self.provider_id == 'xai':
                chat_cls = XaiChat
            else:
                chat_cls = None
            if chat_cls:
                return chat_cls.model_validate(
                    dict(
                        chat_id=chat_id,
                        history=(
                            [dict(role='system', content=self.system_prompt)]
                            if self.system_prompt
                            else []
                        ),
                    )
                )
        raise ValueError(f'Unknown provider: {self.provider_id!r}')

    def swap_conversation(self, chat_id: int, new_conversation_id: int | None) -> bool:
        conversation_dir = os.path.join(config.state_dir, 'conversations', f'chat_{chat_id}')
        os.makedirs(conversation_dir, exist_ok=True)
        if (old_conversation := self.conversation) is not None:
            # save with overwriting
            with NamedTemporaryFile('wb', dir=conversation_dir, delete=False) as tmp_f:
                msgpack.dump(old_conversation.model_dump(), tmp_f)
                os.replace(
                    tmp_f.name,
                    os.path.join(
                        conversation_dir, self._make_conversation_file_name(old_conversation.id)
                    ),
                )
        if new_conversation_id is None:
            # reset
            self.conversation = Conversation(
                id=self.min_unused_conversation_id,
                lm_chat=self._make_chat(),
            )
            self.min_unused_conversation_id += 1
        else:
            # load
            new_conversation_path = os.path.join(
                conversation_dir, self._make_conversation_file_name(new_conversation_id)
            )
            try:
                with open(new_conversation_path, 'rb') as f:
                    data = msgpack.load(f)
                self.conversation = Conversation.model_validate(data)
            except FileNotFoundError:
                return False
            except Exception as exc:
                raise RuntimeError(
                    f'Failed to load the conversation {new_conversation_path!r}'
                ) from exc
        return True

    def remember_conversation_id(self, message: telegram.Message):
        if (c := self.conversation) is not None:
            self.conversation_id_by_message_id[message.message_id] = c.id


class BotData(State):
    pass


# exists simply for type annotations
class Context(CallbackContext[ExtBot, UserData, ChatData, BotData]):
    def __reduce__(self):
        return (self.__class__, ())


def _get_chat_data_with_impersonation(ctx: Context) -> ChatData | None:
    chat_data = ctx.chat_data
    if not chat_data or chat_data.impersonated_user_id is None:
        return chat_data
    return ctx.application.chat_data[chat_data.impersonated_user_id]


def _switch_conversations_impl(
    user_message: telegram.Message, chat_data: ChatData
) -> Iterable[str]:
    conversation = chat_data.conversation
    if (parent_message := user_message.reply_to_message) is not None:
        parent_message_id = parent_message.message_id
        parent_conversation_id = chat_data.conversation_id_by_message_id.get(parent_message_id)
        if parent_conversation_id is not None and (
            not conversation or parent_conversation_id != conversation.id
        ):
            if chat_data.swap_conversation(user_message.chat_id, parent_conversation_id):
                yield f'_Switched to conversation {parent_conversation_id}_'
            else:
                yield '_Unfortunately, this conversation was not saved_'


def switch_conversations(func):
    @wraps(func)
    async def wrapped(update: Update, context: Context, *args, **kwargs):
        assert (message := update.message) is not None
        assert (chat_data := context.chat_data) is not None
        prev_message = message
        for text in _switch_conversations_impl(message, chat_data):
            prev_message = await _reply_or_send(prev_message, text, context)
        chat_data.remember_conversation_id(message)
        return await func(update, context, *args, **kwargs)

    return wrapped


def ensure_convesation_started(func):
    @wraps(func)
    async def wrapped(update: Update, context: Context, *args, **kwargs):
        assert (message := update.message) is not None
        assert (chat_data := context.chat_data) is not None
        if chat_data.conversation is None:
            assert chat_data.swap_conversation(message.chat_id, None)
        return await func(update, context, *args, **kwargs)

    return wrapped


def _split_into_chunks(s: str) -> Iterable[str]:
    if not s:
        return ()
    return split_markdown_into_chunks(  # split evenly
        s, ceil(len(s) / ceil(len(s) / 2048))
    )


def _embed_quote(message: telegram.Message) -> str:
    """Return MarkdownV2-formatted message text/caption, prefixed with a Markdown quote if present.

    - Uses text_markdown_v2_urled or caption_markdown_v2_urled as the base.
    - If the message contains a quote (Message.quote or Message.text_quote), its text is rendered
      as a Markdown quote (each line prefixed with "> ") and placed before the base text.
    """
    base = message.text_markdown_v2_urled or message.caption_markdown_v2_urled or ''
    quote = message.quote
    text = quote.text if quote else None
    if text:
        quote_block = '\n'.join(('> ' + line) if line else '>' for line in text.splitlines())
        return f'{quote_block}\n\n{base}' if base else quote_block
    return base


async def _reply_or_send(
    message_or_chat_id: telegram.Message | int,
    text: str,
    context: Context,
    *,
    disable_notification=False,
) -> telegram.Message:
    assert (chat_data := context.chat_data) is not None
    message = message_or_chat_id
    for chunk in _split_into_chunks(text):
        if not chunk:
            if isinstance(message, telegram.Message):
                logging.warning(
                    'Encountered an empty chunk while handling an update from %r in chat %r',
                    message.from_user,
                    message.chat,
                )
            else:
                logging.warning(
                    'Encountered an empty chunk while sending a message to chat %r',
                    message,
                )
            continue
        match message:
            case telegram.Message():
                do_send_message = message.reply_text  # pyright: ignore[reportAssignmentType]
            case int():
                # this is only possible on the first iteration
                def do_send_message(text, *, _chat_id=message, **kwargs):
                    return context.bot.send_message(chat_id=_chat_id, text=text, **kwargs)
            case _:
                raise TypeError(type(message))
        try:
            message = await do_send_message(
                telegramify_markdown.markdownify(chunk),
                disable_notification=disable_notification,
            )
        except telegram.error.BadRequest as exc:
            logging.warning('Could not send a message as Markdown, retrying as HTML:', exc_info=exc)
            message = await do_send_message(
                html.escape(text), parse_mode=telegram.constants.ParseMode.HTML
            )
        chat_data.remember_conversation_id(message)
    assert isinstance(message, telegram.Message), type(message)
    return message


def send_typing_action(func):
    """Send typing action while running `func`."""

    @wraps(func)
    async def command_func(update, context, *args, **kwargs):
        async def send_action_repeatedly():
            while True:
                try:
                    await context.bot.send_chat_action(
                        chat_id=update.effective_message.chat_id,
                        action=telegram.constants.ChatAction.TYPING,
                    )
                except Exception as exc:
                    logging.warning('Failed to send a "typing" action:', exc_info=exc)
                await aio.sleep(2.5)

        action_sending_task = aio.create_task(send_action_repeatedly())
        try:
            result = await func(update, context, *args, **kwargs)
        finally:
            action_sending_task.cancel()
            try:
                await action_sending_task
            except aio.CancelledError:
                pass
            try:
                await context.bot.send_chat_action(
                    chat_id=update.effective_message.chat_id,
                    action='cancel',
                )
            except Exception as exc:
                logging.warning('Failed to send a "cancel" action:', exc_info=exc)
        return result

    return command_func


def _reset_impl(target_chat_data: ChatData, chat_id: int):
    assert target_chat_data.swap_conversation(chat_id, None)


@send_typing_action
async def reset(update: Update, context: Context) -> None:
    assert (message := update.message) is not None
    assert (target_chat_data := _get_chat_data_with_impersonation(context)) is not None
    _reset_impl(target_chat_data, message.chat_id)
    assert (conversation := target_chat_data.conversation) is not None
    assert (target_chat_id := target_chat_data.chat_id) is not None
    if target_chat_id != message.chat_id:
        await _reply_or_send(
            target_chat_id,
            f"_Started conversation {conversation.id} in response to the bot owner's command_",
            context,
            disable_notification=True,
        )
    await _reply_or_send(message, f'_Started conversation {conversation.id}_', context)


class FileProcessingError(Exception):
    pass


class FileTooLargeError(FileProcessingError):
    file_size: int | None


@retry(OSError, telegram.error.TelegramError)
async def _download_file(file: telegram.File):
    return await file.download_as_bytearray()


async def _process_file(
    document: telegram.Document,
    context: Context,
    text: str,
):
    file = await context.bot.get_file(document.file_id)
    file_size = file.file_size
    if file_size is None or file_size > 2**19:
        exc = FileTooLargeError()
        exc.file_size = file_size
        raise exc
    content = await _download_file(file)
    if _pdf.is_pdf(bytes(content)):
        content = await _pdf_cals.load_pdf(_pdf.Attrs(name=file.file_id), bytes(content))
    else:
        content = content.decode()
    return (
        text.strip()
        + f'<attached-file name={document.file_name!r} mime-type={document.mime_type!r}>'
        + content
        + '</attached-file>'
    )


class _AutoresetTimerInfo(State):
    last_message: telegram.Message
    conversation_id: int


async def _reset_conversation_by_timer(context: Context) -> None:
    assert (job := context.job) is not None
    assert (job_data := job.data) is not None
    assert isinstance(job_data, _AutoresetTimerInfo)
    assert (chat_data := context.chat_data) is not None
    assert (conversation := chat_data.conversation) is not None
    if conversation.id != job_data.conversation_id:
        # the conversation has been reset in the meantime
        return
    last_message = job_data.last_message
    assert chat_data.swap_conversation(last_message.chat_id, None)
    assert (conversation := chat_data.conversation) is not None
    await _reply_or_send(
        last_message,
        f'_Auto-started conversation {conversation.id}_',
        context,
        disable_notification=True,
    )


def _remove_job_if_exists(job_queue: telegram.ext.JobQueue, name: str) -> None:
    for job in job_queue.get_jobs_by_name(name):
        job.schedule_removal()


def _bump_reset_timer(
    chat_id: int,
    last_message: telegram.Message,
    chat_data: ChatData,
    context: Context,
) -> None:
    assert (job_queue := context.job_queue) is not None
    job_name = str(chat_id)
    _remove_job_if_exists(job_queue, job_name)
    assert (conversation := chat_data.conversation) is not None
    job_queue.run_once(
        _reset_conversation_by_timer,
        when=chat_data.autoreset_grace_seconds,
        chat_id=chat_id,
        name=job_name,
        data=_AutoresetTimerInfo(
            last_message=last_message,
            conversation_id=conversation.id,
        ),
    )


@send_typing_action
@switch_conversations
@ensure_convesation_started
async def respond(update: Update, context: Context) -> None:
    assert (message := update.message) is not None
    assert (chat_data := context.chat_data) is not None
    assert (conversation := chat_data.conversation) is not None
    assert (user_data := context.user_data) is not None
    if (user := message.from_user) is not None:
        user_data.user_id = user.id
        user_data.user_name = user.name
    # Build the outgoing message including any quoted snippet as a Markdown quote.
    our_message = _embed_quote(message)
    if (document := message.document) is not None:
        try:
            our_message = await _process_file(document, context, our_message)
        except FileTooLargeError as exc:
            size_mb = round(exc.file_size / 2**20, 3) if exc.file_size else 'an unknown number of'
            await _reply_or_send(
                message,
                f'_Files must not exceed 0.5 MB in size, got {size_mb} MB_',
                context,
            )
            return
        except UnicodeDecodeError:
            await _reply_or_send(
                message,
                "_Failed to process the file, please make sure it's a valid UTF-8 text or a PDF_",
                context,
            )
            return
        except Exception as exc:
            logging.error('Failed to process file:', exc_info=exc)
            return
    is_empty = True
    parent_message = message
    async for their_message in conversation.lm_chat.send_message(
        our_message,
        context,
    ):
        parent_message = await _reply_or_send(
            parent_message,
            their_message.text,
            context,
            disable_notification=not their_message.do_notify,
        )
        _bump_reset_timer(message.chat_id, parent_message, chat_data, context)
        is_empty = False
    if is_empty:
        await _reply_or_send(parent_message, '_The model did not respond_', context)
        logging.warning('The model did not respond')


@send_typing_action
async def prompt(update: Update, context: Context) -> None:
    assert (message := update.message) is not None
    assert (target_chat_data := _get_chat_data_with_impersonation(context)) is not None
    assert (our_message := message.text_markdown_v2_urled) is not None
    target_chat_data.system_prompt = our_message.strip().removeprefix('/prompt').lstrip()
    if target_chat_data.system_prompt:
        await _reply_or_send(
            message, '_Set the system prompt for new and old conversations_', context
        )
    else:
        await _reply_or_send(
            message, '_Cleared the system prompt for new and old conversations_', context
        )


@send_typing_action
async def usage(update: Update, context: Context) -> None:
    assert (message := update.message) is not None
    if (user := update.effective_user) is not None and user.id == config.owner:
        response_text = '_Usage:_\n\n'
        for _user_id, other_user_data in sorted(
            context.application.user_data.items(),
            key=lambda item: -item[1].total_cost_dollars,
        ):
            cost = other_user_data.total_cost_dollars
            if cost:
                response_text += f'{other_user_data.name_with_id}: {cost:.2f}$\n\n'
    else:
        assert (user_data := context.user_data) is not None
        cost = user_data.total_cost_dollars
        response_text = f'_Usage: {cost:.2f}$_'
    await _reply_or_send(message, response_text, context)


@send_typing_action
async def broadcast(update: Update, context: Context) -> None:
    assert (message := update.message) is not None
    text = message.text_markdown_v2_urled
    assert text is not None
    text = text.strip().removeprefix('/broadcast').lstrip()
    if not text:
        await _reply_or_send(message, '_Please provide a message to broadcast_', context)
        return
    tasks = [
        (
            aio.create_task(
                context.bot.send_message(
                    chat_id=user_id,
                    text=text,
                    disable_notification=True,
                    parse_mode=telegram.constants.ParseMode.MARKDOWN_V2,
                )
            ),
            user_data,
        )
        for user_id, user_data in context.application.user_data.items()
        if user_id
    ]
    logging.info('Broadcasting to %r users', len(tasks))
    for task, user_data in tasks:
        try:
            await task
        except Exception as exc:
            if user_data.user_id:
                logging.error(
                    'Failed to send a message to %r:', user_data.name_with_id, exc_info=exc
                )
    await _reply_or_send(
        message, f'_Broadcasted_ {telegramify_markdown.escape_markdown(text)!r}', context
    )


async def post_init(app: telegram.ext.Application) -> None:
    """
    Post initialization hook for the bot.
    """
    await app.bot.set_my_commands(
        [
            telegram.BotCommand('reset', 'Reset the conversation'),
            telegram.BotCommand('prompt', 'Set the initial system prompt for new conversations'),
            telegram.BotCommand('provider', 'Get or set the provider of the language model'),
            telegram.BotCommand('autoreset', 'Set the autoreset grace period in seconds'),
            telegram.BotCommand('usage', 'See the money paid to LLM providers so far'),
        ]
    )


async def handle_error(update: telegram.Update | None, context: Context) -> None:
    assert (exc := context.error) is not None
    tb_string = '```\n' + ''.join(traceback.format_exception(None, exc, exc.__traceback__)) + '```'
    report = 'An exception was raised'
    if update is not None:
        if (message := update.message) is not None:
            try:
                await message.reply_text(
                    text=telegramify_markdown.markdownify(
                        f'_Sorry, there was an error( Please tell @{config.owner_nickname}_'
                    ),
                )
            except Exception as exc_2:
                logging.warning(
                    'Failed to reply to a message while handling an error:', exc_info=exc_2
                )
        report += (
            f' while handling an update from `{update.effective_user}` in `{update.effective_chat}`'
        )
    report += ':'
    logging.warning('%s:\n%s', report, tb_string)
    await context.bot.send_message(
        chat_id=config.owner,
        text=telegramify_markdown.markdownify(report),
    )
    for chunk in _split_into_chunks(tb_string):
        await context.bot.send_message(
            chat_id=config.owner,
            text=telegramify_markdown.markdownify(chunk),
        )


@send_typing_action
async def provider(update: Update, context: Context) -> None:
    assert (message := update.message) is not None
    assert (target_chat_data := _get_chat_data_with_impersonation(context)) is not None
    assert (text := message.text) is not None
    provider = text.lstrip().removeprefix('/provider').strip().lower()
    if not provider:
        available_providers_str = ', '.join(f'`{p}`' for p in _CHAT_CLS_BY_PROVIDER_ID)
        await _reply_or_send(
            message,
            f'_The provider is currently set to `{target_chat_data.provider_id}`, available providers: {available_providers_str}_',
            context,
        )
        return
    if provider not in _CHAT_CLS_BY_PROVIDER_ID:
        await _reply_or_send(message, f'_Unknown provider `{provider}`_', context)
        return
    target_chat_data.provider_id = provider  # pyright: ignore[reportAttributeAccessIssue]
    _reset_impl(target_chat_data, message.chat_id)
    assert (conversation := target_chat_data.conversation) is not None
    await _reply_or_send(
        message,
        f'_Set the provider to `{provider}` and started conversation {conversation.id}_',
        context,
    )


@send_typing_action
async def autoreset(update: Update, context: Context) -> None:
    assert (message := update.message) is not None
    assert (target_chat_data := _get_chat_data_with_impersonation(context)) is not None
    seconds: int = int(4.5 * 60)
    if (text := message.text_markdown_v2_urled) is not None:
        text = text.strip().removeprefix('/autoreset').lstrip()
        if text:
            try:
                seconds = int(text)
            except ValueError:
                await _reply_or_send(
                    message, '_Please provide a valid whole number of seconds_', context
                )
                return
    target_chat_data.autoreset_grace_seconds = seconds
    await _reply_or_send(
        message,
        f'_Set the autoreset grace period to {seconds} seconds_',
        context,
    )


@send_typing_action
async def impersonate(update: Update, context: Context) -> None:
    assert (message := update.message) is not None
    assert (text := message.text_markdown_v2_urled) is not None
    assert (user := message.from_user) is not None
    sender_id = user.id
    assert sender_id == config.owner
    text = text.strip().removeprefix('/impersonate').lstrip()
    if not text:
        context.application.chat_data[sender_id].impersonated_user_id = None
        await _reply_or_send(message, '_Stopped impersonating any user_', context)
        return
    try:
        impersonated_user_id = int(text)
    except ValueError:
        await _reply_or_send(message, '_Please provide a valid user ID_', context)
        return
    context.application.chat_data[sender_id].impersonated_user_id = impersonated_user_id
    await _reply_or_send(message, f'_Impersonating user {impersonated_user_id}_', context)


if __name__ == '__main__':
    logging.getLogger('LiteLLM').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    _validate_tools_for_openai(_BASE_TOOLS)
    args = Args().parse_args()
    config = Config.model_validate(yaml.safe_load(Path(args.config_path).read_bytes()))
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        format='{asctime} {name:>32.32} {levelname:>5}] {message}',
        style='{',
        stream=open(config.log_path, 'a'),
    )
    _pdf_cals = _pdf.Cals(
        base_dir=Path(config.pdf_cals_base_dir),
        mathpix_api_id=config.mathpix_config.api_id,
        mathpix_api_key=config.mathpix_config.api_key,
    )
    app = (
        ApplicationBuilder()
        .token(config.telegram_bot_token)
        .post_init(post_init)
        .persistence(Persistence(config.state_path, ChatData, UserData, BotData))
        .context_types(ContextTypes(user_data=UserData, chat_data=ChatData, bot_data=BotData))
        .defaults(
            defaults=Defaults(
                parse_mode=telegram.constants.ParseMode.MARKDOWN_V2,
                do_quote=True,
            )
        )
        .build()
    )
    DEFAULT_FILTERS = telegram.ext.filters.Chat(chat_id=config.allowed_chat_ids, allow_empty=True)
    app.add_handler(
        CommandHandler(
            'reset',
            reset,
            filters=DEFAULT_FILTERS,
        )  # pyright: ignore[reportArgumentType]
    )
    app.add_handler(
        CommandHandler(
            'prompt',
            prompt,
            filters=DEFAULT_FILTERS,
        )  # pyright: ignore[reportArgumentType]
    )
    app.add_handler(
        CommandHandler(
            'provider',
            provider,
            filters=DEFAULT_FILTERS,
        )  # pyright: ignore[reportArgumentType]
    )
    app.add_handler(
        MessageHandler(
            DEFAULT_FILTERS
            & (telegram.ext.filters.TEXT | telegram.ext.filters.ATTACHMENT)
            & ~telegram.ext.filters.COMMAND,
            respond,
        )  # pyright: ignore[reportArgumentType]
    )
    app.add_handler(
        CommandHandler(
            'usage',
            usage,
            filters=DEFAULT_FILTERS,
        )  # pyright: ignore[reportArgumentType]
    )
    app.add_handler(
        CommandHandler(
            'autoreset',
            autoreset,
            filters=DEFAULT_FILTERS,
        )  # pyright: ignore[reportArgumentType]
    )
    app.add_handler(
        CommandHandler(
            'broadcast',
            broadcast,
            filters=DEFAULT_FILTERS & telegram.ext.filters.Chat(chat_id=config.owner),
        )  # pyright: ignore[reportArgumentType]
    )
    app.add_handler(
        CommandHandler(
            'impersonate',
            impersonate,
            filters=DEFAULT_FILTERS & telegram.ext.filters.Chat(chat_id=config.owner),
        )  # pyright: ignore[reportArgumentType]
    )
    app.add_error_handler(handle_error)  # pyright: ignore[reportArgumentType]
    app.run_polling()
