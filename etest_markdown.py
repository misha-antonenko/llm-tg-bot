import io
from textwrap import dedent
from unittest import main

from expecttest import TestCase

from _markdown import _parse_markdown, _StrSplitter, split_markdown_into_chunks


class TestStrSplitter(TestCase):
    def _split(self, *ss: str, max_chunk_size: int, do_use_repr=True):
        splitter = _StrSplitter(max_chunk_size)
        res = ['']
        for s in ss:
            for subchunk in splitter.split(s):
                if not isinstance(subchunk, str):
                    res.append('')
                else:
                    res[-1] += subchunk
        buf = io.StringIO()
        for s in res:
            self.assertLessEqual(len(s), max_chunk_size)
            print((repr if do_use_repr else str)(s), file=buf)
        return buf.getvalue()

    def test_splitting(self):
        self.assertExpectedInline(
            self._split('abcd', max_chunk_size=3),
            """\
'ab'
'cd'
""",
        )
        md = """
Searches Google for the given query `query`. Returns a list of 10 web page titles together with their
URLs and short snippets (typically under 300 characters in length). The URLs can then be passed
to `fetch_main_text`, for example.
"""
        self.assertExpectedInline(
            self._split(md, max_chunk_size=64),
            """\
'\\nSearches Google for the given query `query`. Returns a list of '
'10 web page titles together with their\\nURLs and short snippets '
'(typically under 300 characters in length). The URLs can then be'
' passed\\nto `fetch_main_text`, for example.\\n'
""",
        )
        self.assertExpectedInline(
            self._split(md, max_chunk_size=32),
            """\
'\\nSearches Google for the given '
'query `query`. Returns a list of'
' 10 web page titles together '
'with their\\nURLs and short '
'snippets (typically under 300 '
'characters in length). The URLs '
'can then be passed\\nto '
'`fetch_main_text`, for example.\\n'
""",
        )
        self.assertExpectedInline(
            self._split(md, max_chunk_size=64),
            """\
'\\nSearches Google for the given query `query`. Returns a list of '
'10 web page titles together with their\\nURLs and short snippets '
'(typically under 300 characters in length). The URLs can then be'
' passed\\nto `fetch_main_text`, for example.\\n'
""",
        )
        md = """
Searches Google for the given query `query`. Returns a list of 10 web page titles together with their
URLs and short snippets (typically under 300 characters in length). The URLs can then be passed
to `fetch_main_text`, for example.

`page_idx` is the pagination index (so that passing 0 will return the first 10 results
that Google knows, passing 1 will return the next 10, etc.). `page_idx` must be in the range
[0, 9].

Here is a quick reminder on the operators Google search supports:
1. `site:example.com` - restricts the search to the given site
2. `filetype:pdf` - restricts the search to the given file type
3. `intitle:word` - restricts the search to pages with the given word in the title
4. `inurl:word` - restricts the search to pages with the given word in the URL
5. `"some phrase"` - restricts the search to pages that contain an exact match for `some phrase`.
    This prohibits Google from including the results for synonyms and permutations, thus
    shrinking the result set severely. Use it only when the results without an exact match
    contain too many irrelevant pages.
"""
        self.assertExpectedInline(
            self._split(md, max_chunk_size=128),
            """\
'\\nSearches Google for the given query `query`. Returns a list of 10 web page titles together with their\\nURLs and short snippets '
'(typically under 300 characters in length). The URLs can then be passed\\nto `fetch_main_text`, for example.\\n\\n`page_idx` is the '
'pagination index (so that passing 0 will return the first 10 results\\nthat Google knows, passing 1 will return the next 10, '
'etc.). `page_idx` must be in the range\\n[0, 9].\\n\\nHere is a quick reminder on the operators Google search supports:\\n1. '
'`site:example.com` - restricts the search to the given site\\n2. `filetype:pdf` - restricts the search to the given file type\\n3. '
'`intitle:word` - restricts the search to pages with the given word in the title\\n4. `inurl:word` - restricts the search to pages '
'with the given word in the URL\\n5. `"some phrase"` - restricts the search to pages that contain an exact match for `some phrase`.'
'\\n    This prohibits Google from including the results for synonyms and permutations, thus\\n    shrinking the result set severely.'
' Use it only when the results without an exact match\\n    contain too many irrelevant pages.\\n'
""",
        )


# @unittest.skip('not ready')
class TestMarkdown(TestCase):
    def _validate(self, chunks: list[str], max_chunk_size: int):
        for chunk in chunks:
            parsed = _parse_markdown(chunk)
            self.assertEqual(type(parsed), list)
            self.assertGreater(len(parsed), 0)
            for item in parsed:
                self.assertEqual(type(item), dict)
            # self.assertLessEqual(len(chunk), 3 * max(12, max_chunk_size))

    def _split(self, md: str, max_chunk_size: int):
        chunks = list(split_markdown_into_chunks(md, max_chunk_size))
        self._validate(chunks, max_chunk_size)
        buf = io.StringIO()
        for chunk in chunks:
            print(repr(chunk), file=buf)
        return buf.getvalue()

    def test_split(self):
        self.assertExpectedInline(
            self._split('abcd', 3),
            """\
'ab'
'cd'
""",
        )
        self.assertExpectedInline(
            self._split('This is a pretty long paragraph', 6),
            """\
'This'
'is a'
'pretty'
'long'
'paragr'
'aph'
""",
        )
        md = """
```cpp
this is a block of code in C++
```
"""
        self.assertExpectedInline(
            self._split(md, 6),
            """\
'```cpp\\nthis \\n```'
'```cpp\\nis a \\n```'
'```cpp\\nblock \\n```'
'```cpp\\nof \\n```'
'```cpp\\ncode \\n```'
'```cpp\\nin C++\\n```'
'```cpp\\n\\n```'
""",
        )
        md = dedent("""
            > this is a
            > loooooong
            > quote by some important prick
        """)
        self.assertExpectedInline(
            self._split(md, 3),
            """\
'> th'
'> is'
'> is'
'> a'
'> loo'
'> ooo'
'> ong'
'>  qu'
'> ote'
'>  by'
'>  so'
'> me'
'> imp'
'> ort'
'> ant'
'>  pr'
'> ick'
""",
        )
        md = dedent("""
            - some rather long list item
            - a shorter one
        """)
        self.assertExpectedInline(
            self._split(md, 3),
            """\
'- so'
'- me'
'- ra'
'- th'
'- er'
'- lo'
'- ng'
'- li'
'- st'
'- it'
'- em\\n- a'
'-'
'- sho'
'- rte'
'- r'
'- one'
""",
        )
        md = dedent("""
            # heading 1
            paragraph 1

            ## heading 2
            paragraph 2

            ### heading 3
        """)
        self.assertExpectedInline(
            self._split(md, 6),
            """\
'# heading 1\\n\\nparagr'
'aph 1\\n\\n## heading 2'
'paragr'
'aph 2\\n\\n### heading 3'
""",
        )
        md = dedent("""
            _italics, and even **bold** ones, are ~~not~~ supported (as well as `code`)_
        """)
        self.assertExpectedInline(
            self._split(md, 6),
            """\
'_italic_'
'_s, and_'
'_ even _'
'_*bold* _'
'_ones, _'
'_are _'
'_~~not~~_'
'_ _'
'_suppor_'
'_ted _'
'_(as _'
'_well _'
'_as _'
'_`code`)_'
""",
        )
        md = dedent("""
            Searches Google for the given query `query`. Returns a list of 10 web page titles together with their
            URLs and short snippets (typically under 300 characters in length). The URLs can then be passed
            to `fetch_main_text`, for example.
        """)
        self.assertExpectedInline(
            self._split(md, 64),
            """\
'Searches Google for the given query `query`. Returns a list of 10'
'web page titles together with their URLs and short snippets'
'(typically under 300 characters in length). The URLs can then be'
'passed to `fetch_main_text`, for example.'
""",
        )


if __name__ == '__main__':
    main()
