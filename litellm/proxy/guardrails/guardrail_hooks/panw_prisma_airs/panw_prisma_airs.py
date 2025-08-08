#!/usr/bin/env python3
"""
PANW Prisma AIRS Built-in Guardrail for LiteLLM

"""

import os
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Literal, Optional, Type, Union, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from litellm.types.utils import ModelResponseStream

class PanwPrismaAirsHandler(CustomGuardrail):
    """
    LiteLLM Built-in Guardrail for Palo Alto Networks Prisma AI Runtime Security (AIRS).

    This guardrail scans prompts and responses using the PANW Prisma AIRS API to detect
    malicious content, injection attempts, and policy violations.

    Configuration:
        guardrail_name: Name of the guardrail instance
        api_key: PANW Prisma AIRS API key
        api_base: PANW Prisma AIRS API endpoint
        profile_name: PANW Prisma AIRS security profile name
        incremental_scan_chunk_size: Size of text chunks to scan incrementally (0 for full response scan, default 0)
        disable_streaming_check: If True, disables streaming checks and passes through all chunks, only run post_call_success_hook. defaults to False
        default_on: Whether to enable by default
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: str,
        api_base: str,
        profile_name: str,
        incremental_scan_chunk_size: int = 0,
        default_on: bool = True,
        disable_streaming_check: bool = False,
        **kwargs,
    ):
        """Initialize PANW Prisma AIRS guardrail handler."""

        # Initialize parent CustomGuardrail
        super().__init__(guardrail_name=guardrail_name, default_on=default_on, **kwargs)

        # Store configuration
        self.api_key = api_key or os.getenv("PANW_PRISMA_AIRS_API_KEY")
        self.api_base = (
            api_base
            or os.getenv("PANW_PRISMA_AIRS_API_BASE")
            or "https://service.api.aisecurity.paloaltonetworks.com"
        )
        self.profile_name = profile_name
        self.incremental_scan_chunk_size = incremental_scan_chunk_size
        self.disable_streaming_check = disable_streaming_check  

        verbose_proxy_logger.info(
            f"Initialized PANW Prisma AIRS Guardrail: {guardrail_name}, incremental_scan_chunk_size: {incremental_scan_chunk_size}, disable_streaming_check: {disable_streaming_check}"
        )

    def _extract_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text content from messages array."""
        if not isinstance(messages, list) or not messages:
            return ""

        # Find the last user message
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if not content:
                continue

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                return self._extract_text_from_content_list(content)

        return ""

    def _extract_text_from_content_list(
        self, content_list: List[Dict[str, Any]]
    ) -> str:
        """Extract text from content list format."""
        text_parts = [
            part.get("text", "")
            for part in content_list
            if isinstance(part, dict)
            and part.get("type") == "text"
            and part.get("text")
        ]
        return " ".join(text_parts) if text_parts else ""

    def _extract_response_text(self, response: ModelResponse) -> str:
        """Extract text from LLM response."""
        try:
            from litellm.types.utils import Choices

            if (
                hasattr(response, "choices")
                and response.choices
                and len(response.choices) > 0
                and hasattr(response.choices[0], "message")
            ):
                return cast(Choices, response.choices[0]).message.content or ""
        except (AttributeError, IndexError):
            verbose_proxy_logger.error(
                "PANW Prisma AIRS: Error extracting response text"
            )
        return ""

    def _extract_text_from_chunk(self, chunk: "ModelResponseStream") -> str:
        """Extract text content from a streaming chunk."""
        try:
            if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                    return choice.delta.content or ""
        except (AttributeError, IndexError):
            verbose_proxy_logger.debug("PANW Prisma AIRS: No content in chunk")
        return ""

    async def _call_panw_api(
        self,
        content: str,
        is_response: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call PANW Prisma AIRS API to scan content."""

        if not content.strip():
            return {"action": "allow", "category": "empty"}

        # Build request payload
        transaction_id = (
            f"litellm-{'resp' if is_response else 'req'}-{uuid.uuid4().hex[:8]}"
        )

        payload = {
            "tr_id": transaction_id,
            "ai_profile": {"profile_name": self.profile_name},
            "metadata": {
                "app_user": (
                    metadata.get("user", "litellm_user") if metadata else "litellm_user"
                ),
                "ai_model": metadata.get("model", "unknown") if metadata else "unknown",
                "source": "litellm_builtin_guardrail",
            },
            "contents": [{"response" if is_response else "prompt": content}],
        }

        if is_response:
            payload["metadata"]["is_response"] = True  # type: ignore[index]

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-pan-token": self.api_key,
        }

        try:
            # Use LiteLLM's async HTTP client
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )

            response = await async_client.post(
                f"{self.api_base}/v1/scan/sync/request",
                headers=headers,
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()

            result = response.json()

            # Validate response format
            if "action" not in result:
                verbose_proxy_logger.error(
                    f"PANW Prisma AIRS: Invalid API response format: {result}"
                )
                return {"action": "block", "category": "api_error"}

            verbose_proxy_logger.debug(
                f"PANW Prisma AIRS: Scan result - Action: {result.get('action')}, Category: {result.get('category', 'unknown')}"
            )
            return result

        except Exception as e:
            verbose_proxy_logger.error(f"PANW Prisma AIRS: API call failed: {str(e)}")
            return {"action": "block", "category": "api_error"}

    def _build_error_detail(
        self, scan_result: Dict[str, Any], is_response: bool = False
    ) -> Dict[str, Any]:
        """Build enhanced error detail with scan information."""
        action_type = "Response" if is_response else "Prompt"
        code_suffix = "_response_blocked" if is_response else "_blocked"
        detection_key = "response_detected" if is_response else "prompt_detected"

        category = scan_result.get("category", "unknown")
        error_msg = f"{action_type} blocked by PANW Prisma AI Security policy (Category: {category})"

        error_detail = {
            "error": {
                "message": error_msg,
                "type": "guardrail_violation",
                "code": f"panw_prisma_airs{code_suffix}",
                "guardrail": self.guardrail_name,
                "category": category,
            }
        }

        # Add optional fields if present
        optional_fields = [
            "scan_id",
            "report_id",
            "profile_name",
            "profile_id",
            "tr_id",
        ]
        for field in optional_fields:
            if scan_result.get(field):
                error_detail["error"][field] = scan_result[field]

        # Add detection details
        if scan_result.get(detection_key):
            error_detail["error"][detection_key] = scan_result[detection_key]

        return error_detail

    def _check_scan_result(
        self, scan_result: Dict[str, Any], is_response: bool = False
    ) -> None:
        """Check scan result and raise HTTPException if content should be blocked."""
        action = scan_result.get("action", "block")
        category = scan_result.get("category", "unknown")

        if action == "allow":
            verbose_proxy_logger.info(
                f"PANW Prisma AIRS: {'Response' if is_response else 'Prompt'} allowed (Category: {category})"
            )
        else:
            error_detail = self._build_error_detail(scan_result, is_response=is_response)
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS: {error_detail['error']['message']}"
            )
            raise HTTPException(status_code=400, detail=error_detail)

    def _create_error_chunk(self, assembled_model_response: ModelResponse, error_detail: Dict[str, Any]) -> "ModelResponseStream":
        """Create an error chunk to terminate the stream."""
        from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta
        
        return ModelResponseStream(
            id=assembled_model_response.id if hasattr(assembled_model_response, 'id') else 'chatcmpl-error',
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=f"\n\n[BLOCKED BY PANW PRISMA AIRS]: Error: 400 {error_detail}"
                    ),
                    finish_reason="stop"
                )
            ],
            model=assembled_model_response.model if hasattr(assembled_model_response, 'model') else 'unknown',
            object="chat.completion.chunk"
        )

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: Dict[str, Any],
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-call hook to scan user prompts before sending to LLM.

        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.info("PANW Prisma AIRS: Running pre-call prompt scan")

        # Extract prompt text from messages
        messages = data.get("messages", [])
        prompt_text = self._extract_text_from_messages(messages)

        if not prompt_text:
            verbose_proxy_logger.warning(
                "PANW Prisma AIRS: No user prompt found in request"
            )
            return None

        # Prepare metadata
        metadata = {
            "user": data.get("user", "litellm_user"),
            "model": data.get("model", "unknown"),
        }

        # Scan prompt with PANW Prisma AIRS
        scan_result = await self._call_panw_api(
            content=prompt_text, is_response=False, metadata=metadata
        )

        # Check scan result and raise exception if needed
        self._check_scan_result(scan_result, is_response=False)

        return None

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: ModelResponse,
    ) -> ModelResponse:
        """
        Post-call hook to scan LLM responses before returning to user.

        Raises HTTPException if response should be blocked.
        """
        verbose_proxy_logger.info("PANW Prisma AIRS: Running post-call response scan")

        # Extract response text
        response_text = self._extract_response_text(response)

        if not response_text:
            verbose_proxy_logger.warning(
                "PANW Prisma AIRS: No response content found to scan"
            )
            return response

        # Prepare metadata
        metadata = {
            "user": data.get("user", "litellm_user"),
            "model": data.get("model", "unknown"),
        }

        # Scan response with PANW Prisma AIRS
        scan_result = await self._call_panw_api(
            content=response_text, is_response=True, metadata=metadata
        )

        # Check scan result and raise exception if needed
        self._check_scan_result(scan_result, is_response=True)

        return response

    @log_guardrail_information
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: Dict[str, Any],
    ) -> AsyncGenerator["ModelResponseStream", None]:
        """
        Process streaming response chunks for PANW Prisma AIRS scanning.

        Supports both incremental scanning (configurable interval) and complete response scanning.
        If incremental_scan_chunk_size > 0, checks content at regular intervals and only yields chunks after verification.
        If incremental_scan_chunk_size = 0, only checks the complete response at the end.
        If disable_streaming_check is True, passes through all chunks without any scanning.
        """
        # Import here to avoid circular imports
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import TextCompletionResponse

        # Check if streaming check is disabled, if so, pass through all chunks directly
        if getattr(self, 'disable_streaming_check', False):
            verbose_proxy_logger.info("PANW Prisma AIRS: Streaming check disabled, passing through all chunks")
            async for chunk in response:
                yield chunk
            return

        verbose_proxy_logger.info(
            f"PANW Prisma AIRS: Running streaming response scan with incremental_scan_chunk_size={self.incremental_scan_chunk_size}"
        )

        # Prepare metadata
        metadata = {
            "user": request_data.get("user", "litellm_user"),
            "model": request_data.get("model", "unknown"),
        }

        # If incremental checking is disabled (interval = 0), collect all chunks first
        if self.incremental_scan_chunk_size == 0:
            verbose_proxy_logger.info("PANW Prisma AIRS: Incremental checking disabled, collecting all chunks for final scan")
            
            # Collect all chunks to process them together
            all_chunks: List["ModelResponseStream"] = []
            try:
                async for chunk in response:
                    all_chunks.append(chunk)
                    # Do not yield chunks immediately, wait for check results
            except Exception as e:
                # If there's an error collecting chunks, just pass through
                verbose_proxy_logger.error(f"PANW Prisma AIRS: Error collecting chunks: {str(e)}")
                return

            # Assemble the complete response from chunks for final scanning
            assembled_model_response: Optional[
                Union[ModelResponse, TextCompletionResponse]
            ] = stream_chunk_builder(
                chunks=all_chunks,
            )

            if isinstance(assembled_model_response, (type(None), TextCompletionResponse)):
                # If we can't assemble a ModelResponse or it's a text completion, 
                # yield all chunks (fail open approach)
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: Could not assemble ModelResponse from chunks, yielding all chunks"
                )
                for chunk in all_chunks:
                    yield chunk
                return

            # Extract response text for final scanning
            response_text = self._extract_response_text(assembled_model_response)
            if response_text:
                verbose_proxy_logger.info(
                    f"PANW Prisma AIRS: Performing final scan on complete response ({len(response_text)} characters)"
                )
                
                try:
                    # Scan response with PANW Prisma AIRS
                    scan_result = await self._call_panw_api(
                        content=response_text, is_response=True, metadata=metadata
                    )
                    
                    action = scan_result.get("action", "block")
                    category = scan_result.get("category", "unknown")

                    if action != "allow":
                        # Content is blocked - don't yield the chunks, return error instead
                        error_detail = self._build_error_detail(scan_result, is_response=True)
                        verbose_proxy_logger.warning(
                            f"PANW Prisma AIRS: Final scan blocked complete response - {error_detail['error']['message']}"
                        )
                        # Create and yield error chunk instead of original content
                        error_chunk = self._create_error_chunk(assembled_model_response, error_detail)
                        yield error_chunk
                        return
                    else:
                        verbose_proxy_logger.info(
                            f"PANW Prisma AIRS: Final scan passed for complete response (Category: {category})"
                        )
                        # Content is safe - yield all chunks
                        for chunk in all_chunks:
                            yield chunk
                        return
                        
                except Exception as e:
                    verbose_proxy_logger.error(f"PANW Prisma AIRS: Error during final scanning: {str(e)}")
                    # On scanning error, yield all chunks (fail open approach)
                    verbose_proxy_logger.warning("PANW Prisma AIRS: Yielding all chunks due to scanning error")
                    for chunk in all_chunks:
                        yield chunk
                    return
            else:
                # No content to scan, yield all chunks
                for chunk in all_chunks:
                    yield chunk
            
            return

        # Incremental checking is enabled (interval > 0)
        verbose_proxy_logger.info(f"PANW Prisma AIRS: Incremental checking enabled with interval {self.incremental_scan_chunk_size}")
        
        # Collect chunks and accumulated text for incremental checking
        all_chunks: List["ModelResponseStream"] = []
        accumulated_text = ""
        last_yielded_position = 0  # Track how much content we've already yielded to user
        pending_chunks: List["ModelResponseStream"] = []  # Chunks waiting for approval

        try:
            async for chunk in response:
                all_chunks.append(chunk)
                pending_chunks.append(chunk)
                
                # Extract text from current chunk
                chunk_text = self._extract_text_from_chunk(chunk)
                if chunk_text:
                    accumulated_text += chunk_text
                
                # Check if we've accumulated enough content for incremental check
                if len(accumulated_text) - last_yielded_position >= self.incremental_scan_chunk_size:
                    
                    verbose_proxy_logger.info(
                        f"PANW Prisma AIRS: Performing incremental check at position {len(accumulated_text)} (pending approval for {len(accumulated_text) - last_yielded_position} characters)"
                    )
                    
                    try:
                        # Scan the accumulated text up to this point
                        scan_result = await self._call_panw_api(
                            content=accumulated_text, is_response=True, metadata=metadata
                        )
                        
                        action = scan_result.get("action", "block")
                        category = scan_result.get("category", "unknown")

                        if action != "allow":
                            # Content is blocked - stop streaming and return error
                            error_detail = self._build_error_detail(scan_result, is_response=True)
                            verbose_proxy_logger.warning(
                                f"PANW Prisma AIRS: Incremental check blocked content at position {len(accumulated_text)} - {error_detail['error']['message']}"
                            )
                            
                            # Create error chunk using the last valid chunk format
                            if all_chunks:
                                partial_assembled_response = stream_chunk_builder(chunks=all_chunks)
                                if partial_assembled_response and not isinstance(partial_assembled_response, TextCompletionResponse):
                                    error_chunk = self._create_error_chunk(partial_assembled_response, error_detail)
                                    yield error_chunk
                            return
                        else:
                            verbose_proxy_logger.debug(
                                f"PANW Prisma AIRS: Incremental check passed at position {len(accumulated_text)} (Category: {category})"
                            )
                            
                            # Content is safe - yield all pending chunks
                            for pending_chunk in pending_chunks:
                                yield pending_chunk
                            
                            # Update tracking variables
                            last_yielded_position = len(accumulated_text)
                            pending_chunks.clear()
                
                    except Exception as e:
                        verbose_proxy_logger.error(f"PANW Prisma AIRS: Error during incremental scanning: {str(e)}")
                        # On scanning error, yield pending chunks (fail open approach)
                        verbose_proxy_logger.warning("PANW Prisma AIRS: Yielding pending chunks due to scanning error")
                        for pending_chunk in pending_chunks:
                            yield pending_chunk
                        last_yielded_position = len(accumulated_text)
                        pending_chunks.clear()
                
        except Exception as e:
            # If there's an error collecting chunks, log and stop
            verbose_proxy_logger.error(f"PANW Prisma AIRS: Error during streaming: {str(e)}")
            return

        # Handle any remaining pending chunks after the loop ends
        if pending_chunks and accumulated_text:
            remaining_chars = len(accumulated_text) - last_yielded_position
            if remaining_chars > 0:
                # Check if streaming check is disabled before performing final incremental check
                if getattr(self, 'disable_streaming_check', False):
                    verbose_proxy_logger.info("PANW Prisma AIRS: Streaming check disabled, yielding remaining pending chunks without final check")
                    for pending_chunk in pending_chunks:
                        yield pending_chunk
                    return
                
                verbose_proxy_logger.info(
                    f"PANW Prisma AIRS: Performing final incremental check on remaining {remaining_chars} characters"
                )
                
                try:
                    # Scan the complete accumulated text
                    scan_result = await self._call_panw_api(
                        content=accumulated_text, is_response=True, metadata=metadata
                    )
                    
                    action = scan_result.get("action", "block")
                    category = scan_result.get("category", "unknown")

                    if action != "allow":
                        # Content is blocked - return error instead of pending chunks
                        error_detail = self._build_error_detail(scan_result, is_response=True)
                        verbose_proxy_logger.warning(
                            f"PANW Prisma AIRS: Final incremental check blocked remaining content - {error_detail['error']['message']}"
                        )
                        
                        # Create error chunk
                        if all_chunks:
                            final_assembled_response = stream_chunk_builder(chunks=all_chunks)
                            if final_assembled_response and not isinstance(final_assembled_response, TextCompletionResponse):
                                error_chunk = self._create_error_chunk(final_assembled_response, error_detail)
                                yield error_chunk
                        return
                    else:
                        verbose_proxy_logger.info(
                            f"PANW Prisma AIRS: Final incremental check passed (Category: {category})"
                        )
                        
                        # Content is safe - yield remaining pending chunks
                        for pending_chunk in pending_chunks:
                            yield pending_chunk
                        
                except Exception as e:
                    verbose_proxy_logger.error(f"PANW Prisma AIRS: Error during final incremental scanning: {str(e)}")
                    # On scanning error, yield pending chunks (fail open approach)
                    verbose_proxy_logger.warning("PANW Prisma AIRS: Yielding remaining chunks due to scanning error")
                    for pending_chunk in pending_chunks:
                        yield pending_chunk
            else:
                # No remaining content to check, yield any pending chunks
                for pending_chunk in pending_chunks:
                    yield pending_chunk

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Dict[str, Any]]:
        """
        Moderation hook to scan user prompts for content moderation during call processing.

        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.info("PANW Prisma AIRS: Running moderation prompt scan")

        # Extract prompt text from messages
        messages = data.get("messages", [])
        prompt_text = self._extract_text_from_messages(messages)

        if not prompt_text:
            verbose_proxy_logger.warning(
                "PANW Prisma AIRS: No user prompt found in request for moderation"
            )
            return data

        # Prepare metadata
        metadata = {
            "user": data.get("user", "litellm_user"),
            "model": data.get("model", "unknown"),
        }

        # Scan prompt with PANW Prisma AIRS
        scan_result = await self._call_panw_api(
            content=prompt_text, is_response=False, metadata=metadata
        )

        # Check scan result and raise exception if needed
        self._check_scan_result(scan_result, is_response=False)

        return data

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.panw_prisma_airs import (
            PanwPrismaAirsGuardrailConfigModel,
        )

        return PanwPrismaAirsGuardrailConfigModel