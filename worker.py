"""Taskiq Worker 启动入口脚本。

用法：
    `python worker.py --workers 2 --log-level INFO`

说明：
- 该脚本会固定加载 broker：`backend.infrastructure.task_queue.broker:broker`
- 并加载任务模块：`backend.worker.tasks`
- 其余参数将原样转发给 `taskiq worker` 子命令（需放在 broker 之前的参数）。
"""

import subprocess
import sys


def main() -> None:
    """启动 Taskiq Worker 进程。"""

    cmd = [
        sys.executable,
        "-m",
        "taskiq",
        "worker",
        *sys.argv[1:],
        "backend.infrastructure.task_queue.broker:broker",
        "backend.worker.tasks",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
