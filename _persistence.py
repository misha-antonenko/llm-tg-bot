import logging
import os
import time
from tempfile import NamedTemporaryFile
from typing import Generic, TypeVar

import msgpack
import yaml
from pydantic import BaseModel
from telegram.ext import BasePersistence, PersistenceInput

UD = TypeVar('UD', bound=BaseModel)
"""Type of the user data for a single user.
"""
CD = TypeVar('CD', bound=BaseModel)
"""Type of the chat data for a single user.
"""
BD = TypeVar('BD', bound=BaseModel)
"""Type of the bot data.
"""


class Persistence(BasePersistence, Generic[UD, CD, BD]):
    def _load(self):
        try:
            with open(self._state_path, 'rb') as f:
                try:
                    self._state = msgpack.load(f, strict_map_key=False)
                except Exception:
                    f.seek(0)
                    self._state = yaml.safe_load(f)
        except FileNotFoundError:
            self._state = {}
        assert isinstance(self._state, dict)

        chat_data = self._state.get('chat_data')
        self._chat_data = (
            {
                int(k): self._chat_data_cls.model_validate(v, context=dict(chat_id=int(k)))
                for k, v in chat_data.items()
            }
            if chat_data is not None
            else None
        )

        user_data = self._state.get('user_data')
        self._user_data = (
            {
                int(k): self._user_data_cls.model_validate(v, context=dict(user_id=int(k)))
                for k, v in user_data.items()
            }
            if user_data is not None
            else None
        )

        bot_data = self._state.get('bot_data')
        self._bot_data = (
            self._bot_data_cls.model_validate(bot_data) if bot_data is not None else None
        )

        self._conversations: dict[str, dict] | None = self._state.get('conversations')
        self._callback_data: object = self._state.get('callback_data')

    def __init__(
        self,
        state_path: str,
        chat_data_cls: type[BaseModel],
        user_data_cls: type[BaseModel],
        bot_data_cls: type[BaseModel],
        store_data: PersistenceInput | None = None,
        update_interval: float = 60.0,
    ):
        super().__init__(store_data, update_interval)

        self._chat_data_cls = chat_data_cls
        self._user_data_cls = user_data_cls
        self._bot_data_cls = bot_data_cls

        self._state_path = state_path
        self._load()

        self._last_flush_ts = time.monotonic() - self._update_interval

    async def get_chat_data(self):
        if self._chat_data is None:
            self._chat_data = {}
        return self._chat_data

    async def get_user_data(self):
        if self._user_data is None:
            self._user_data = {}
        return self._user_data

    async def get_bot_data(self):
        if self._bot_data is None:
            self._bot_data = self._bot_data_cls()
        return self._bot_data

    async def get_callback_data(self):
        return self._callback_data

    async def get_conversations(self, name: str):
        if self._conversations is None:
            self._conversations = {}
        return self._conversations.setdefault(name, {})

    async def update_chat_data(self, chat_id: int, data: CD):
        if self._chat_data is None:
            self._chat_data = {}
        self._chat_data[chat_id] = data
        await self._flush_if_needed()

    async def update_user_data(self, user_id: int, data: UD):
        if self._user_data is None:
            self._user_data = {}
        self._user_data[user_id] = data
        await self._flush_if_needed()

    async def update_bot_data(self, data: BD):
        self._bot_data = data
        await self._flush_if_needed()

    async def update_callback_data(self, data):
        self._callback_data = data
        await self._flush_if_needed()

    async def update_conversation(self, name: str, key, new_state) -> None:
        if self._conversations is None:
            self._conversations = {}
        self._conversations.setdefault(name, {})[key] = new_state
        await self._flush_if_needed()

    async def drop_chat_data(self, chat_id: int) -> None:
        if self._chat_data is not None:
            self._chat_data.pop(chat_id, None)

    async def drop_user_data(self, user_id: int) -> None:
        if self._user_data is not None:
            self._user_data.pop(user_id, None)

    async def refresh_user_data(self, user_id: int, user_data: UD) -> None:
        pass

    async def refresh_chat_data(self, chat_id: int, chat_data: CD) -> None:
        chat_data.chat_id = chat_id  # TODO(atm)

    async def refresh_bot_data(self, bot_data: BD) -> None:
        pass

    def _dump(self):
        res = {}
        if self._chat_data is not None:
            res['chat_data'] = {k: v.model_dump() for k, v in self._chat_data.items()}
        if self._user_data is not None:
            res['user_data'] = {k: v.model_dump() for k, v in self._user_data.items()}
        if self._bot_data is not None:
            res['bot_data'] = self._bot_data.model_dump()
        if self._callback_data is not None:
            res['callback_data'] = self._callback_data
        if self._conversations is not None:
            res['conversations'] = self._conversations
        logging.debug('dumping %r', res)

        os.makedirs('tmp', exist_ok=True)
        with NamedTemporaryFile('wb', dir='tmp', delete=False) as tmp_f:
            # json.dump(res, tmp_f, indent=2, ensure_ascii=False)
            msgpack.dump(res, tmp_f)
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            os.replace(tmp_f.name, self._state_path)

    async def _flush_if_needed(self):
        if time.monotonic() - self._last_flush_ts >= self._update_interval:
            await self.flush()

    async def flush(self):
        logging.debug(msg='Flushing')
        self._dump()
        logging.debug('Flushed')
        self._last_flush_ts = time.monotonic()
