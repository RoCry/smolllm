"""Adapter for SmolLLM to be used with smolagents"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union

from smolagents.models import Model, ChatMessage, ChatMessageToolCall, ChatMessageToolCallDefinition, parse_tool_args_if_needed
from smolagents import Tool

from .log import logger
from .core import ask_llm
from .async_util import run_async

class SmolllmModel(Model):
    """Model adapter for SmolLLM to be used with smolagents
    
    Parameters:
        model_id (`str`):
            The model identifier to use with SmolLLM (e.g. "gemini/gemini-2.0-flash").
        api_key (`str`, *optional*):
            The API key to use for authentication. If not provided, will use environment variables.
        base_url (`str`, *optional*):
            The base URL of the provider API.
        timeout (`float`, *optional*, defaults to 60.0):
            Timeout for the API request, in seconds.
        **kwargs:
            Additional keyword arguments to pass to the underlying API.
    """
    
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        
    def _extract_text_from_content(self, content: Union[str, List[Dict[str, Any]]]) -> str:
        """Extract text from content which can be a string or a list of content parts"""
        if isinstance(content, str):
            return content
        
        # If content is a list of content parts, extract the text from each part
        text_parts = []
        for part in content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            # Currently ignoring other content types like images
            
        return " ".join(text_parts)
        
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> tuple[str, Optional[str]]:
        """Convert smolagents message format to SmolLLM prompt format"""
        system_prompt = None
        user_prompts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Extract text from the content
            text_content = self._extract_text_from_content(content)
            
            if role == "system":
                system_prompt = text_content
            elif role == "user":
                user_prompts.append(text_content)
            elif role == "assistant":
                # For multi-turn conversations, we'd need to handle this
                # For now we just focus on the user messages
                pass
        
        # Join all user messages into a single prompt
        prompt = "\n".join(user_prompts)
        return prompt, system_prompt
    
    @run_async
    async def _call_llm(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Call the LLM with the given prompt and system prompt"""
        return await ask_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model_id,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
    
    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages using SmolLLM and return the model's response.
        
        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.
        
        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        prompt, system_prompt = self._format_messages_to_prompt(messages)
        
        # Call the LLM with the formatted prompts
        response_text = self._call_llm(prompt, system_prompt)
        
        # Rough token counting estimate
        self.last_input_token_count = len(prompt.split()) 
        self.last_output_token_count = len(response_text.split())
        
        # Handle tool calls if needed
        if tools_to_call_from:
            try:
                # Try to find a tool call in the response
                # Looking for a JSON structure in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    tool_json = response_text[start_idx:end_idx+1]
                    tool_data = json.loads(tool_json)
                    
                    # Extract tool name and arguments based on common formats
                    # Adjust this logic based on actual model output format
                    tool_name = tool_data.get("name") or tool_data.get("action") or tool_data.get("tool")
                    tool_args = tool_data.get("arguments") or tool_data.get("action_input") or tool_data.get("params")
                    
                    if tool_name:
                        return ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id=str(uuid.uuid4()),
                                    type="function",
                                    function=ChatMessageToolCallDefinition(
                                        name=tool_name,
                                        arguments=tool_args
                                    )
                                )
                            ],
                            raw=response_text
                        )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse tool call: {e}")
        
        # Return regular message response
        return ChatMessage(
            role="assistant",
            content=response_text,
            raw=response_text
        )
    