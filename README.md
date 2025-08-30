# llm-telegram-bot

this is a Telegram UI for large language model APIs

# how to run

1. `cp example_config.yaml config.yaml`, then configure the users and API keys in this `config.yaml`
2. [install uv](https://docs.astral.sh/uv/)
3. run `uv run __main__.py`

# why

compared to the subscription-based services, the generative model APIs often offer
1. a higher and more stable generation quality
2. more customization points (choose system prompts and tools)

there is a variety of ways to use these APIs already. there is OpenWebUI, if you need a
browser-based UI. this bot offers a Telegram-based UI, which
1. is easier to host
2. has super-reliable conversation persistence
3. may be more handy for some users
4. does not need proxies or VPNs for users to access your service under totalitarian regimes

# features

- [x] persist conversations between restarts
- [x] multiple conversations with ability to return to any past one by replying to a message in it
- [x] multiple providers:
    - [x] OpenAI
    - [x] Anthropic
    - [x] Google
    - [x] xAI
- [x] PDF input (the PDFs are converted into text via Mathpix)
- [x] text file input
- [x] web search
- [x] fetching web pages and PDFs
- [x] WolframAlpha
- [x] system prompts
- [x] memory
- [ ] image input
- [ ] voice input
- [ ] code execution
- [ ] ready-made providers' tools (like the built-in search or code execution by OpenAI or Google)
- [ ] let the user choose any model and provider supported by LiteLLM

# contributions

contributions are always welcome!
