# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Global text-only LLM manager (singleton pattern).

This client is intended for high-quality language-only workloads such as
conversation, summarization and planning. Vision workloads should use
``GlobalVLMClient`` so that lightweight multimodal models can be selected
separately.
"""

import asyncio
import concurrent.futures
import json
import threading
from typing import Any, Dict, Optional

from opencontext.config.global_config import get_config
from opencontext.llm.llm_client import LLMClient, LLMType
from opencontext.tools.tools_executor import ToolsExecutor
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GlobalLLMClient:
    """Global language-model manager (singleton pattern)."""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._llm_client: Optional[LLMClient] = None
                    self._tools_executor = ToolsExecutor()
                    GlobalLLMClient._initialized = True
                    self._auto_initialized = False

    @classmethod
    def get_instance(cls) -> "GlobalLLMClient":
        instance = cls()
        if not instance._auto_initialized and instance._llm_client is None:
            instance._auto_initialize()
        return instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    def _auto_initialize(self):
        if self._auto_initialized:
            return
        try:
            llm_config = get_config("llm_model")
            if not llm_config:
                logger.warning("No llm config found in llm_model")
                self._auto_initialized = True
                return

            self._llm_client = LLMClient(llm_type=LLMType.CHAT, config=llm_config)
            logger.info("GlobalLLMClient auto-initialized successfully")
            self._auto_initialized = True
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"GlobalLLMClient auto-initialization failed: {exc}")
            self._auto_initialized = True

    def is_initialized(self) -> bool:
        return self._llm_client is not None

    def reinitialize(self):
        with self._lock:
            try:
                llm_config = get_config("llm_model")
                if not llm_config:
                    logger.error("No llm config found during reinitialize")
                    raise ValueError("No llm config found")
                new_client = LLMClient(llm_type=LLMType.CHAT, config=llm_config)
                self._llm_client = new_client
                logger.info("GlobalLLMClient reinitialized successfully")
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"Failed to reinitialize LLM client: {exc}")
                return False
            return True

    def generate_with_messages(
        self, messages: list, enable_executor: bool = True, max_calls: int = 5, **kwargs
    ):
        response = self._llm_client.generate_with_messages(messages, **kwargs)
        call_count = 0
        while enable_executor:
            call_count += 1
            if call_count > max_calls:
                logger.warning(
                    "Reached maximum tool call limit (%s), stopping further calls", max_calls
                )
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "System notice: Maximum tool call limit "
                            f"({max_calls}) reached. Cannot execute more tool calls. "
                            "Please answer the user's question directly without attempting more tool calls."
                        ),
                    }
                )
                response = self._llm_client.generate_with_messages(messages, **kwargs)
                break
            message = response.choices[0].message
            if not message.tool_calls:
                break
            messages.append(message)
            tool_calls = message.tool_calls
            tool_call_info = []
            for tc in tool_calls:
                function_name = tc.function.name
                function_args = parse_json_from_response(tc.function.arguments)
                tool_call_info.append((tc.id, function_name, function_args))
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_tool = {
                    executor.submit(self._tools_executor.run, function_name, function_args): (
                        tool_id,
                        function_name,
                    )
                    for tool_id, function_name, function_args in tool_call_info
                }
                for future in concurrent.futures.as_completed(future_to_tool):
                    tool_id, function_name = future_to_tool[future]
                    try:
                        content = future.result()
                        results.append((tool_id, function_name, content))
                    except Exception:  # pylint: disable=broad-except
                        results.append((tool_id, function_name, "failed"))
            for tool_id, function_name, content in results:
                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(content),
                        "tool_call_id": tool_id,
                    }
                )
            response = self._llm_client.generate_with_messages(messages, **kwargs)

        message = response.choices[0].message
        return message.content

    async def generate_with_messages_async(
        self, messages: list, enable_executor: bool = True, max_calls: int = 5, **kwargs
    ):
        response = await self._llm_client.generate_with_messages_async(messages, **kwargs)
        call_count = 0
        while enable_executor:
            call_count += 1
            if call_count > max_calls:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "System notice: Maximum tool call limit "
                            f"({max_calls}) reached. Cannot execute more tool calls. "
                            "Please answer the user's question directly without attempting more tool calls."
                        ),
                    }
                )
                response = await self._llm_client.generate_with_messages_async(
                    messages, **kwargs
                )
                break
            message = response.choices[0].message
            if not message.tool_calls:
                break
            messages.append(message)
            tool_calls = message.tool_calls

            tasks = []
            for tc in tool_calls:
                try:
                    function_args = parse_json_from_response(tc.function.arguments)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(f"Tool call args parse error: {exc}")
                    continue
                if function_args is None:
                    continue
                tasks.append(
                    (
                        tc.id,
                        tc.function.name,
                        asyncio.create_task(
                            self._tools_executor.run_async(tc.function.name, function_args)
                        ),
                    )
                )

            if tasks:
                results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
                for idx, result in enumerate(results):
                    tool_id, function_name, _ = tasks[idx]
                    if isinstance(result, Exception):
                        content = "failed"
                        logger.error(
                            "Tool %s execution failed in async LLM client: %s", function_name, result
                        )
                    else:
                        content = result
                    messages.append(
                        {
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(content),
                            "tool_call_id": tool_id,
                        }
                    )
            response = await self._llm_client.generate_with_messages_async(messages, **kwargs)

        message = response.choices[0].message
        return message.content

    async def generate_for_agent_async(self, messages: list, tools: list = None, **kwargs):
        response = await self._llm_client.generate_with_messages_async(
            messages, tools=tools, **kwargs
        )
        return response

    async def generate_stream_for_agent(self, messages: list, tools: list = None, **kwargs):
        async for chunk in self._llm_client._openai_chat_completion_stream_async(  # type: ignore[attr-defined]
            messages, tools=tools, **kwargs
        ):
            yield chunk

    async def execute_tool_async(self, tool_call):
        function_name = tool_call.function.name
        function_args = parse_json_from_response(tool_call.function.arguments)

        if function_args is None:
            logger.error(
                "Failed to parse arguments for %s: %s", function_name, tool_call.function.arguments
            )
            return {"error": f"Failed to parse arguments for {function_name}"}

        try:
            result = await self._tools_executor.run_async(function_name, function_args)
            logger.info("Tool %s executed successfully", function_name)
            return result
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Tool %s execution failed: %s", function_name, exc)
            return {"error": str(exc)}


def is_initialized() -> bool:
    return GlobalLLMClient.get_instance()._auto_initialized


def generate_with_messages(
    messages: list, enable_executor: bool = True, max_calls: int = 5, **kwargs
):
    return GlobalLLMClient.get_instance().generate_with_messages(
        messages, enable_executor, max_calls, **kwargs
    )


async def generate_with_messages_async(
    messages: list, enable_executor: bool = True, max_calls: int = 5, **kwargs
):
    return await GlobalLLMClient.get_instance().generate_with_messages_async(
        messages, enable_executor, max_calls, **kwargs
    )


async def generate_for_agent_async(messages: list, tools: list = None, **kwargs):
    return await GlobalLLMClient.get_instance().generate_for_agent_async(messages, tools, **kwargs)


def generate_stream_for_agent(messages: list, tools: list = None, **kwargs):
    return GlobalLLMClient.get_instance().generate_stream_for_agent(messages, tools, **kwargs)
