import re
from collections.abc import Iterable
from copy import copy
from typing import Any

import mistune as mt
from mistune.core import BlockState
from mistune.renderers.markdown import MarkdownRenderer
from pydantic import BaseModel


class _Split(BaseModel):
    pass


_WHITESPACE_RE = re.compile(r'(\s+)')


class _StrSplitter:
    def __init__(self, max_chunk_size: int):
        self._max_chunk_size = max_chunk_size
        self._chunk_size = 0
        self._is_previous_subchunk_final = False
        self._subchunk = ''

    def _append_short_token(self, s: str):
        if self._chunk_size + len(s) > self._max_chunk_size:
            if self._subchunk:
                if self._is_previous_subchunk_final:
                    yield _Split()
                yield self._subchunk
            self._is_previous_subchunk_final = True
            self._subchunk = s
            self._chunk_size = len(s)
        else:
            self._subchunk += s
            self._chunk_size += len(s)

    def split(self, s: str):
        """
        The sum of token lengths between two splits, or between a split and stream beginning / end,
        does not surpass `max_chunk_size`.
        All tokens that are yielded from `split(s)` give `s` when concatenated.
        The sequence `split(s)` may never end with a split, but may begin with a split.
        """
        for token in re.split(_WHITESPACE_RE, s):
            assert isinstance(token, str), token
            max_chunk_size = (
                # for prettiness
                min(self._max_chunk_size, round(len(token) ** 0.5))
                if len(token) > self._max_chunk_size
                else self._max_chunk_size
            )
            while True:
                head, tail = token[:max_chunk_size], token[max_chunk_size:]
                yield from self._append_short_token(head)
                if not tail:
                    break
                token = tail
        if self._subchunk:
            if self._is_previous_subchunk_final:
                yield _Split()
            yield self._subchunk
            if self._chunk_size + len(self._subchunk) == self._max_chunk_size:
                self._is_previous_subchunk_final = True
            else:
                self._is_previous_subchunk_final = False
            self._subchunk = ''


class _MarkdownSplitter:
    def __init__(self, max_chunk_size: int):
        self._content_splitter = _StrSplitter(max_chunk_size)

    def _split_children(self, item: dict[str, Any]) -> Iterable[_Split | dict[str, Any]]:
        new_children = []

        def flush():
            nonlocal new_children
            if new_children:
                res = copy(item)
                res['children'] = new_children
                yield res
                new_children = []

        children = item['children']
        for i, child in enumerate(children):
            if child['type'] == 'softbreak':
                continue
            if i < len(children) - 1:
                next_child_ty = children[i + 1]['type']
                if next_child_ty == 'softbreak' and 'raw' in child:
                    # TODO(atm): we still want to split `child` if there is no `raw`
                    child['raw'] += '\n'
            for chunk in self.split(child):
                if isinstance(chunk, _Split):
                    yield from flush()
                    yield chunk
                else:
                    new_children.append(chunk)
        yield from flush()

    def split(self, item: dict[str, Any]) -> Iterable[_Split | dict[str, Any]]:
        assert isinstance(item, dict)
        if content := item.get('raw'):  # a leaf
            for chunk in self._content_splitter.split(content):
                if isinstance(chunk, _Split):
                    yield chunk
                    continue
                res = copy(item)
                res['raw'] = chunk
                yield res
        else:
            ty = item['type']
            if ty in ('heading',):
                yield item  # don't split; TODO(atm)
            elif 'children' not in item:  # a leaf; most likely whitespace
                yield item
            else:
                try:
                    yield from self._split_children(item)
                except Exception as exc:
                    raise ValueError(f'Failed to split the children of {item}') from exc


class _MarkdownRenderer(MarkdownRenderer):
    def emphasis(self, token: dict[str, Any], state: BlockState) -> str:
        return '_' + self.render_children(token, state) + '_'

    def strong(self, token: dict[str, Any], state: BlockState) -> str:
        return '*' + self.render_children(token, state) + '*'

    def thematic_break(self, _token: dict[str, Any], _state: BlockState) -> str:
        return '---\n\n'

    def strikethrough(_renderer, token, _state):
        [child] = token['children']
        return f'~~{child["raw"]}~~'


_renderer = _MarkdownRenderer()


def _render_markdown(tokens) -> str:
    return _renderer(tokens, BlockState())


class _InlineParser(mt.InlineParser):
    def parse_codespan(self, m: re.Match, state: mt.InlineState) -> int:
        marker = m.group(0)
        # require same marker with same length at end

        pattern = re.compile(r'(.*?[^`])' + marker + r'(?!`)', re.S)

        pos = m.end()
        m = pattern.match(state.src, pos)
        if m:
            end_pos = m.end()
            code = m.group(1)
            # Line endings are treated like spaces
            code = code.replace('\n', ' ')
            if len(code.strip()):
                if code.startswith(' ') and code.endswith(' '):
                    code = code[1:-1]
            # the only difference from the original method is that we don't escape `code`
            # in the following line
            state.append_token({'type': 'codespan', 'raw': code})
            return end_pos
        else:
            state.append_token({'type': 'text', 'raw': marker})
            return pos


def _make_markdown_parser() -> mt.Markdown:
    """Create a Markdown instance based on the given condition.

    :param escape: Boolean. If using html renderer, escape html.
    :param hard_wrap: Boolean. Break every new line into ``<br>``.
    :param renderer: renderer instance, default is HTMLRenderer.
    :param plugins: List of plugins.

    This method is used when you want to re-use a Markdown instance::

        markdown = create_markdown(
            escape=False,
            hard_wrap=True,
        )
        # re-use markdown function
        markdown('.... your text ...')
    """
    inline = _InlineParser(hard_wrap=False)
    plugins = [mt.plugins.import_plugin(n) for n in ['strikethrough']]
    return mt.Markdown(renderer=None, inline=inline, plugins=plugins)


_parse_markdown = _make_markdown_parser()


def split_markdown_into_chunks(s: str, max_chunk_size: int) -> Iterable[str]:
    tokens = _parse_markdown(s)
    splitter = _MarkdownSplitter(max_chunk_size)
    token_chunk = []

    def flush():
        nonlocal token_chunk
        if token_chunk:
            yield _render_markdown(token_chunk).strip()
            token_chunk = []

    for token in tokens:
        for item in splitter.split(token):
            if isinstance(item, _Split):
                yield from flush()
            else:
                token_chunk.append(item)
    yield from flush()
