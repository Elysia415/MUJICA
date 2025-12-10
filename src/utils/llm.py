from openai import OpenAI
import os
import json
from typing import List, Dict, Optional, Any

def get_llm_client():
    """
    Returns an initialized OpenAI client.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        return None
    return OpenAI(api_key=api_key)

def chat(messages: List[Dict[str, str]], 
         model: str = "gpt-4",
         temperature: float = 0.2,
         client: Optional[OpenAI] = None) -> str:
    """
    通用聊天完成函数
    
    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
        model: 模型名称
        temperature: 温度参数
        client: OpenAI 客户端实例，如果为 None 则自动创建
    
    Returns:
        LLM 返回的文本内容
    """
    if client is None:
        client = get_llm_client()
        if client is None:
            raise ValueError("Failed to initialize OpenAI client")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat: {e}")
        return ""
