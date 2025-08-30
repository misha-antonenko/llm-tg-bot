import pytest

from _markdown import split_markdown_into_chunks


@pytest.mark.parametrize(
    'markdown, max_chunk_size, expected_chunks',
    [
        # Test case 1: Empty string
        ('', 100, ['']),
        # Test case 2: Single paragraph shorter than max_chunk_size
        ('This is a short paragraph.', 100, ['This is a short paragraph.']),
        # Test case 3: Single paragraph longer than max_chunk_size
        (
            'This is a longer paragraph that exceeds the maximum chunk size.',
            20,
            ['This is a longer', 'paragraph that', 'exceeds the maximum', 'chunk size.'],
        ),
        # Test case 4: Multiple paragraphs
        (
            'Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.',
            50,
            ['Paragraph 1.', 'Paragraph 2.', 'Paragraph 3.'],
        ),
        # Test case 5: Headers
        (
            '# Header 1\n\nContent 1\n\n## Header 2\n\nContent 2',
            50,
            ['# Header 1\n\nContent 1\n\n## Header 2', 'Content 2'],
        ),
        # Test case 6: Lists
        ('- Item 1\n- Item 2\n- Item 3', 50, ['- Item 1\n- Item 2\n- Item 3']),
        # Test case 7: Code blocks
        (
            '```python\ndef function():\n    pass\n```\n\nText after code block.',
            50,
            ['```python\ndef function():\n    pass\n```', 'Text after code block.'],
        ),
        # Test case 8: Blockquotes
        (
            '> This is a blockquote\n> It spans multiple lines\n\nText after blockquote.',
            50,
            ['> This is a blockquote\n> It spans multiple lines', 'Text after blockquote.'],
        ),
        # Test case 9: Mixed content
        (
            '# Header\n\nParagraph 1.\n\n- List item 1\n- List item 2\n\n```\nCode block\n```\n\n> Blockquote',
            50,
            [
                '# Header',
                'Paragraph 1.',
                '- List item 1\n- List item 2',
                '```\nCode block\n```',
                '> Blockquote',
            ],
        ),
        # Test case 10: Long words
        (
            'Supercalifragilisticexpialidocious is a very long word.',
            10,
            ['Supercalif\n', 'ragilistic\n', 'expialidoc\n', 'ious is a\n', 'very long\n', 'word.\n'],
        ),
    ],
)
def test_split_markdown_into_chunks(markdown, max_chunk_size, expected_chunks):
    result = list(split_markdown_into_chunks(markdown, max_chunk_size))
    assert result == expected_chunks



