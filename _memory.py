import os
import shutil
from pathlib import Path


class Memory:
    def __init__(self, base: Path):
        self._base = base

    def _validate_key(self, key: str):
        path = Path(self._base, key)
        for part in path.parts:
            if part == '/':
                continue
            if not part.isidentifier():
                raise ValueError(
                    f'each path segment of `key` must be a valid identifier in Python, got {part!r}'
                )
        return path

    def set_memo(self, key: str, value: str):
        path = self._validate_key(key)
        os.makedirs(path.parent, exist_ok=True)
        if value:
            with open(path, 'x') as f:
                f.writelines((value,))
        else:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            # remove empty parent directories
            for p in path.parents:
                if p == self._base or next(p.iterdir(), None):
                    break
                p.rmdir()

    def get_memo(self, key: str) -> str | None:
        path = self._validate_key(key)
        if not path.exists():
            return None
        if path.is_dir():
            raise ValueError(f'{path} is a directory, not a file')
        with open(path) as f:
            return f.read()

    def append_to_memo(self, key: str, value: str):
        path = self._validate_key(key)
        os.makedirs(path.parent, exist_ok=True)
        with open(path, 'a') as f:
            f.writelines((value,))

    def list_directory(self, path: str) -> list[dict[str, str | bool]]:
        return [dict(name=p.name, is_dir=p.is_dir()) for p in self._validate_key(path).iterdir()]

    def glob_memos(self, pattern: str) -> list[str]:
        if pattern.endswith('**'):
            pattern += '/*'
        return [
            str(p.relative_to(self._base))
            for p in self._base.glob(pattern, case_sensitive=False)
            if p.is_file()
        ]
