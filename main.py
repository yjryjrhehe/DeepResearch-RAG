"""
本地启动入口（开发用）。

推荐启动方式：
1) `uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --reload`
2) 或直接执行：`python main.py`
"""

import uvicorn


def main() -> None:
    uvicorn.run("backend.api.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
