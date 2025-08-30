"""
CALS is for content-addressable local storage
"""

import asyncio as aio
import hashlib
import io
import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import MathpixPDFLoader
from pydantic import BaseModel
from pypdf import PdfReader

from _retries import retry


class Attrs(BaseModel):
    name: str


def is_pdf(content: bytes) -> bool:
    return content.startswith(b'%PDF-')


class Cals:
    _MAX_N_PAGES = 64

    def __init__(
        self,
        base_dir: Path,
        mathpix_api_id: str,
        mathpix_api_key: str,
    ):
        self._base_dir = base_dir
        self._mathpix_api_id = mathpix_api_id
        self._mathpix_api_key = mathpix_api_key

        os.makedirs(self._base_dir / 'attrs', exist_ok=True)
        os.makedirs(self._base_dir / 'pdf', exist_ok=True)
        os.makedirs(self._base_dir / 'md', exist_ok=True)

    def _convert_pdf_to_md(self, pdf_path: Path) -> str:
        loader = MathpixPDFLoader(
            pdf_path.as_posix(),
            mathpix_api_id=self._mathpix_api_id,
            mathpix_api_key=self._mathpix_api_key,
            extra_request_data=dict(
                enable_tables_fallback=True,
            ),
        )
        data = loader.load()
        return data[0].page_content

    @classmethod
    def _check_size_limit(cls, content: bytes) -> None:
        reader = PdfReader(io.BytesIO(content))
        if (n_pages := len(reader.pages)) > cls._MAX_N_PAGES:
            raise ValueError(
                f'sending a PDF of {n_pages} > {cls._MAX_N_PAGES} pages to Mathpix is '
                'expensive and probably unjustified. You can object to this'
            )

    def _load_pdf_impl(self, attrs: Attrs, content: bytes) -> str:
        # TODO(atm): this method uses blocking IO
        h = hashlib.sha256(content).hexdigest()
        attrs_path = self._base_dir / 'attrs' / h
        pdf_path = self._base_dir / 'pdf' / h
        md_path = self._base_dir / 'md' / h

        if attrs_path.exists():
            loaded_attrs = Attrs.model_validate_json(attrs_path.read_bytes())
            if loaded_attrs == attrs:
                return md_path.read_text()

        self._check_size_limit(content)

        pdf_path.write_bytes(content)
        md = self._convert_pdf_to_md(pdf_path)
        md_path.write_text(md)
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_attrs:
            temp_attrs.write(Attrs(name=attrs.name).model_dump_json())
        os.replace(temp_attrs.name, attrs_path.as_posix())
        return md

    @retry(IOError)
    async def load_pdf(self, attrs: Attrs, content: bytes) -> str:
        return await aio.to_thread(self._load_pdf_impl, attrs, content)
