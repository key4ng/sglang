"""
Tests for the refactored completions serving handler
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sglang.srt.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    ErrorResponse,
)
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


@pytest.fixture
def mock_tokenizer_manager():
    """Create a mock tokenizer manager"""
    manager = Mock(spec=TokenizerManager)

    # Mock tokenizer
    manager.tokenizer = Mock()
    manager.tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
    manager.tokenizer.decode = Mock(return_value="decoded text")
    manager.tokenizer.bos_token_id = 1

    # Mock model config
    manager.model_config = Mock()
    manager.model_config.is_multimodal = False

    # Mock server args
    manager.server_args = Mock()
    manager.server_args.enable_cache_report = False

    # Mock generation
    manager.generate_request = AsyncMock()
    manager.create_abort_task = Mock(return_value=None)

    return manager


@pytest.fixture
def serving_completion(mock_tokenizer_manager):
    """Create a OpenAIServingCompletion instance"""
    return OpenAIServingCompletion(mock_tokenizer_manager)


class TestPromptHandling:
    """Test different prompt types and formats from adapter.py"""

    def test_single_string_prompt(self, serving_completion):
        """Test handling single string prompt"""
        request = CompletionRequest(
            model="test-model", prompt="Hello world", max_tokens=100
        )

        adapted_request, _ = serving_completion._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.text == "Hello world"

    def test_single_token_ids_prompt(self, serving_completion):
        """Test handling single token IDs prompt"""
        request = CompletionRequest(
            model="test-model", prompt=[1, 2, 3, 4], max_tokens=100
        )

        adapted_request, _ = serving_completion._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.input_ids == [1, 2, 3, 4]

    def test_completion_template_handling(self, serving_completion):
        """Test completion template processing"""
        request = CompletionRequest(
            model="test-model",
            prompt="def hello():",
            suffix="return 'world'",
            max_tokens=100,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_completions.is_completion_template_defined",
            return_value=True,
        ):
            with patch(
                "sglang.srt.entrypoints.openai.serving_completions.generate_completion_prompt_from_request",
                return_value="processed_prompt",
            ):
                adapted_request, _ = serving_completion._convert_to_internal_request(
                    [request], ["test-id"]
                )

                assert adapted_request.text == "processed_prompt"


class TestEchoHandling:
    """Test echo functionality from adapter.py"""

    def test_echo_with_string_prompt_streaming(self, serving_completion):
        """Test echo handling with string prompt in streaming"""
        request = CompletionRequest(
            model="test-model", prompt="Hello", max_tokens=100, echo=True
        )

        # Test _get_echo_text method
        echo_text = serving_completion._get_echo_text(request, 0)
        assert echo_text == "Hello"

    def test_echo_with_list_of_strings_streaming(self, serving_completion):
        """Test echo handling with list of strings in streaming"""
        request = CompletionRequest(
            model="test-model",
            prompt=["Hello", "World"],
            max_tokens=100,
            echo=True,
            n=1,
        )

        echo_text = serving_completion._get_echo_text(request, 0)
        assert echo_text == "Hello"

        echo_text = serving_completion._get_echo_text(request, 1)
        assert echo_text == "World"

    def test_echo_with_token_ids_streaming(self, serving_completion):
        """Test echo handling with token IDs in streaming"""
        request = CompletionRequest(
            model="test-model", prompt=[1, 2, 3], max_tokens=100, echo=True
        )

        serving_completion.tokenizer_manager.tokenizer.decode.return_value = (
            "decoded_prompt"
        )
        echo_text = serving_completion._get_echo_text(request, 0)
        assert echo_text == "decoded_prompt"

    def test_echo_with_multiple_token_ids_streaming(self, serving_completion):
        """Test echo handling with multiple token ID prompts in streaming"""
        request = CompletionRequest(
            model="test-model", prompt=[[1, 2], [3, 4]], max_tokens=100, echo=True, n=1
        )

        serving_completion.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        echo_text = serving_completion._get_echo_text(request, 0)
        assert echo_text == "decoded"

    def test_prepare_echo_prompts_non_streaming(self, serving_completion):
        """Test prepare echo prompts for non-streaming response"""
        # Test with single string
        request = CompletionRequest(model="test-model", prompt="Hello", echo=True)

        echo_prompts = serving_completion._prepare_echo_prompts(request)
        assert echo_prompts == ["Hello"]

        # Test with list of strings
        request = CompletionRequest(
            model="test-model", prompt=["Hello", "World"], echo=True
        )

        echo_prompts = serving_completion._prepare_echo_prompts(request)
        assert echo_prompts == ["Hello", "World"]

        # Test with token IDs
        request = CompletionRequest(model="test-model", prompt=[1, 2, 3], echo=True)

        serving_completion.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        echo_prompts = serving_completion._prepare_echo_prompts(request)
        assert echo_prompts == ["decoded"]


class TestHiddenStates:
    """Test hidden states functionality"""

    def test_hidden_states_request_conversion_single(self, serving_completion):
        """Test request conversion with return_hidden_states=True for single request"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            return_hidden_states=True,
        )

        adapted_request, _ = serving_completion._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.return_hidden_states is True

    def test_hidden_states_request_conversion_multiple(self, serving_completion):
        """Test request conversion with return_hidden_states=True for multiple requests"""
        requests = [
            CompletionRequest(
                model="test-model",
                prompt="Hello",
                return_hidden_states=True,
            ),
            CompletionRequest(
                model="test-model",
                prompt="World",
                return_hidden_states=False,
            ),
        ]

        adapted_request, _ = serving_completion._convert_to_internal_request(
            requests, ["test-id-1", "test-id-2"]
        )

        assert adapted_request.return_hidden_states == [True, False]

    def test_hidden_states_non_streaming_response(self, serving_completion):
        """Test hidden states in non-streaming response"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            return_hidden_states=True,
        )

        ret = [{
            "text": "Test response",
            "meta_info": {
                "id": "cmpl-test",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "hidden_states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # Mock hidden states
            },
        }]

        response = serving_completion._build_completion_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.hidden_states == [0.4, 0.5, 0.6]  # Should return last token's hidden states

    def test_hidden_states_non_streaming_response_no_hidden_states(self, serving_completion):
        """Test response when return_hidden_states=False"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            return_hidden_states=False,
        )

        ret = [{
            "text": "Test response",
            "meta_info": {
                "id": "cmpl-test",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
            },
        }]

        response = serving_completion._build_completion_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.hidden_states is None

    @pytest.mark.asyncio
    async def test_hidden_states_streaming_response(self, serving_completion):
        """Test hidden states in streaming response"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            return_hidden_states=True,
            stream=True,
        )

        # Mock the generate_request to return hidden states
        async def mock_generate():
            yield {
                "text": "Test",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": None,
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "input_token_logprobs": None,
                    "input_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3]],
                },
                "index": 0,
            }
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "input_token_logprobs": None,
                    "input_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                },
                "index": 0,
            }

        serving_completion.tokenizer_manager.generate_request = Mock(return_value=mock_generate())

        adapted_request, _ = serving_completion._convert_to_internal_request(
            [request], ["test-id"]
        )

        mock_raw_request = Mock()
        response = await serving_completion._handle_streaming_request(
            adapted_request, request, mock_raw_request
        )

        # Collect all chunks from the streaming response
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode())

        # Check that hidden states are included in the response
        hidden_states_found = False
        for chunk in chunks:
            if "hidden_states" in chunk and "data:" in chunk:
                hidden_states_found = True
                break

        assert hidden_states_found, "Hidden states should be present in streaming response"

    def test_hidden_states_with_echo_non_streaming(self, serving_completion):
        """Test hidden states with echo enabled in non-streaming response"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_hidden_states=True,
            echo=True,
        )

        ret = [{
            "text": "world",
            "meta_info": {
                "id": "cmpl-test",
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "input_token_logprobs": [],
                "input_top_logprobs": None,
                "hidden_states": [[0.1, 0.2], [0.3, 0.4]],
            },
        }]

        response = serving_completion._build_completion_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.text == "Hello" + "world"  # Echo + completion
        assert choice.hidden_states == [0.3, 0.4]  # Last token's hidden states

    def test_hidden_states_multiple_choices(self, serving_completion):
        """Test hidden states with multiple choices (n > 1)"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_hidden_states=True,
            n=2,
        )

        ret = [
            {
                "text": "world",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2], [0.3, 0.4]],
                },
            },
            {
                "text": "universe",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 5,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.5, 0.6], [0.7, 0.8]],
                },
            }
        ]

        response = serving_completion._build_completion_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 2
        assert response.choices[0].hidden_states == [0.3, 0.4]  # Last token for choice 0
        assert response.choices[1].hidden_states == [0.7, 0.8]  # Last token for choice 1

    def test_hidden_states_empty_list(self, serving_completion):
        """Test handling of empty hidden states list"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_hidden_states=True,
        )

        ret = [{
            "text": "world",
            "meta_info": {
                "id": "cmpl-test",
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "hidden_states": [],  # Empty hidden states
            },
        }]

        response = serving_completion._build_completion_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.hidden_states == []

    def test_hidden_states_single_token(self, serving_completion):
        """Test handling of hidden states with single token"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_hidden_states=True,
        )

        ret = [{
            "text": "world",
            "meta_info": {
                "id": "cmpl-test",
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "hidden_states": [[0.1, 0.2, 0.3]],  # Single token hidden states
            },
        }]

        response = serving_completion._build_completion_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.hidden_states == []  # Should return empty list for single token

    def test_hidden_states_list_request_handling(self, serving_completion):
        """Test hidden states with list of requests"""
        request1 = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_hidden_states=True,
        )
        
        request2 = CompletionRequest(
            model="test-model",
            prompt="World",
            return_hidden_states=False,
        )

        ret1 = [{
            "text": "response1",
            "meta_info": {
                "id": "cmpl-test",
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "hidden_states": [[0.1, 0.2], [0.3, 0.4]],
            },
        }]
        
        ret2 = [{
            "text": "response2",
            "meta_info": {
                "id": "cmpl-test",
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "hidden_states": [[0.5, 0.6], [0.7, 0.8]],
            },
        }]

        # Test first request with return_hidden_states=True
        response1 = serving_completion._build_completion_response(
            request1, ret1, 1234567890
        )

        assert len(response1.choices) == 1
        assert response1.choices[0].hidden_states == [0.3, 0.4]

        # Test second request with return_hidden_states=False
        response2 = serving_completion._build_completion_response(
            request2, ret2, 1234567890
        )

        assert len(response2.choices) == 1
        assert response2.choices[0].hidden_states is None

    def test_hidden_states_token_ids_prompt(self, serving_completion):
        """Test hidden states with token IDs as prompt"""
        request = CompletionRequest(
            model="test-model",
            prompt=[1, 2, 3, 4],
            return_hidden_states=True,
        )

        adapted_request, _ = serving_completion._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.input_ids == [1, 2, 3, 4]
        assert adapted_request.return_hidden_states is True

    @pytest.mark.asyncio 
    async def test_hidden_states_streaming_with_echo(self, serving_completion):
        """Test hidden states in streaming response with echo enabled"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            return_hidden_states=True,
            echo=True,
            stream=True,
        )

        # Mock the generate_request to return hidden states
        async def mock_generate():
            yield {
                "text": " world",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": [],
                    "input_token_logprobs": [[0.9, 1, "Hello"]],
                    "input_top_logprobs": [{}],
                    "hidden_states": [[0.1, 0.2], [0.3, 0.4]],
                },
                "index": 0,
            }

        serving_completion.tokenizer_manager.generate_request = Mock(return_value=mock_generate())

        adapted_request, _ = serving_completion._convert_to_internal_request(
            [request], ["test-id"]
        )

        mock_raw_request = Mock()
        response = await serving_completion._handle_streaming_request(
            adapted_request, request, mock_raw_request
        )

        # Collect all chunks from the streaming response
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode())

        # Should contain both echo text and hidden states
        echo_found = False
        hidden_states_found = False
        
        for chunk in chunks:
            if "Hello" in chunk and "data:" in chunk:
                echo_found = True
            if "hidden_states" in chunk and "data:" in chunk:
                hidden_states_found = True

        assert echo_found, "Echo text should be present in streaming response"
        assert hidden_states_found, "Hidden states should be present in streaming response"
