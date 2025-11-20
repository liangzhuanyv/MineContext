# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Global reranker client placeholder.

Allows configuring a dedicated reranker model that can be invoked through
OpenAI-compatible APIs. Currently it exposes a light wrapper around
:class:`LLMClient` so downstream components can fetch a ready-to-use client
without duplicating initialization code.
"""

import threading
from typing import Optional

from opencontext.config.global_config import get_config
from opencontext.llm.llm_client import LLMClient, LLMType
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GlobalRerankerClient:
    """Singleton wrapper for reranker model access."""

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
                    self._client: Optional[LLMClient] = None
                    GlobalRerankerClient._initialized = True
                    self._auto_initialized = False

    @classmethod
    def get_instance(cls) -> "GlobalRerankerClient":
        instance = cls()
        if not instance._auto_initialized and instance._client is None:
            instance._auto_initialize()
        return instance

    def _auto_initialize(self):
        if self._auto_initialized:
            return
        try:
            reranker_config = get_config("reranker_model")
            if not reranker_config:
                logger.info("No reranker model configured")
                self._auto_initialized = True
                return
            self._client = LLMClient(llm_type=LLMType.CHAT, config=reranker_config)
            logger.info("GlobalRerankerClient auto-initialized successfully")
            self._auto_initialized = True
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"GlobalRerankerClient auto-init failed: {exc}")
            self._auto_initialized = True

    def is_initialized(self) -> bool:
        return self._client is not None

    def get_client(self) -> Optional[LLMClient]:
        return self._client

    def reinitialize(self) -> bool:
        with self._lock:
            try:
                reranker_config = get_config("reranker_model")
                if not reranker_config:
                    self._client = None
                    logger.info("Cleared reranker client because config was empty")
                    return True
                self._client = LLMClient(llm_type=LLMType.CHAT, config=reranker_config)
                logger.info("GlobalRerankerClient reinitialized successfully")
                return True
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(f"Failed to reinitialize reranker client: {exc}")
                return False


def is_initialized() -> bool:
    return GlobalRerankerClient.get_instance()._auto_initialized
