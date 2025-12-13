from openai import OpenAI
import os
import hashlib
from typing import List, Dict, Optional

from src.utils.env import load_env


def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _fake_embedding(text: str, *, dim: int = 384) -> list:
    """
    离线/测试用的确定性 embedding（不依赖外部服务）。
    - 仅用于无 API Key 或希望可复现的单元测试场景
    - 维度可通过 MUJICA_FAKE_EMBEDDING_DIM 调整
    """
    text = (text or "").encode("utf-8")
    dim = max(16, int(dim))

    # 生成足够多的 bytes
    need = dim * 4
    buf = b""
    counter = 0
    while len(buf) < need:
        buf += hashlib.sha256(text + str(counter).encode("utf-8")).digest()
        counter += 1

    vec = []
    for i in range(dim):
        b = buf[i * 4 : (i + 1) * 4]
        u = int.from_bytes(b, "little", signed=False)
        # 映射到 [0,1)
        vec.append((u % 1_000_000) / 1_000_000.0)
    return vec

def get_llm_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """
    Returns an initialized OpenAI client.
    Args:
        api_key: User-provided API key. If None, falls back to env var.
        base_url: User-provided Base URL. If None, falls back to env var.
    """
    load_env()
    # 1. Determine API Key
    final_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
    
    if not final_api_key:
        print("Warning: API Key not found (neither provided nor in env).")
        return None
    
    # 2. Determine Base URL
    final_base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL")
    
    return OpenAI(api_key=final_api_key, base_url=final_base_url)

def get_embedding(text: str, model="text-embedding-3-small", 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None) -> list:
    """
    Generates vector embedding for the given text.
    """
    load_env()
    if _env_truthy("MUJICA_FAKE_EMBEDDINGS"):
        dim = int(os.getenv("MUJICA_FAKE_EMBEDDING_DIM", "384"))
        return _fake_embedding(text, dim=dim)

    client = get_llm_client(api_key=api_key, base_url=base_url)
    if not client:
        return []
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def get_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[list]:
    """
    批量生成 embedding（比逐条请求更快/更省）。
    失败时会返回与输入等长的空向量列表。
    """
    load_env()
    if not texts:
        return []

    if _env_truthy("MUJICA_FAKE_EMBEDDINGS"):
        dim = int(os.getenv("MUJICA_FAKE_EMBEDDING_DIM", "384"))
        return [_fake_embedding(t, dim=dim) for t in texts]

    client = get_llm_client(api_key=api_key, base_url=base_url)
    if not client:
        return [[] for _ in texts]

    try:
        cleaned = [(t or "").replace("\n", " ") for t in texts]
        resp = client.embeddings.create(input=cleaned, model=model)
        # OpenAI 返回顺序与输入一致（每条包含 index）
        out = [None] * len(cleaned)
        for item in resp.data:
            out[item.index] = item.embedding
        return [v if v is not None else [] for v in out]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return [[] for _ in texts]

def chat(messages: List[Dict[str, str]], 
         model: str = "gpt-4o",
         temperature: float = 0.2,
         client: Optional[OpenAI] = None) -> str:
    """
    Generic chat wrapper.
    """
    # Client should be passed in, but if not, logic inside get_llm_client handles env fallback
    if client is None:
        client = get_llm_client()
        if client is None:
            return ""
    
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
