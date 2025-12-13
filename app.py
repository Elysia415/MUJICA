"""
MUJICA Streamlit 入口。

推荐本地启动方式：
    streamlit run ui/app.py

在 HuggingFace Spaces（readme 顶部 frontmatter 指定 app_file=app.py）中，
也可以直接使用本文件作为入口：streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path，方便 `import src.*`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.env import load_env


def main() -> None:
    load_env()
    from ui.app import main as ui_main

    ui_main()


if __name__ == "__main__":
    # 避免用户误用 `python app.py` 直接运行导致 session_state 报错
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

        if get_script_run_ctx() is None:
            print("这是一个 Streamlit 应用，请使用：streamlit run ui/app.py")
        else:
            main()
    except Exception:
        # 若 Streamlit 内部 API 变更，则尽力运行（通常用于 streamlit run）
        main()
