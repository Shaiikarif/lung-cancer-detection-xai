"""Entrypoint script to start the PulmoAI Flask web server."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.webapp.app import create_app
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("run_webapp")


def main():
    parser = argparse.ArgumentParser(description="Start PulmoAI Web Interface")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--host", type=str, default=None, help="Host address (e.g. 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port number (e.g. 5000)")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg["webapp"]["host"]
    port = args.port or cfg["webapp"]["port"]

    app = create_app(args.config)
    logger.info("Starting PulmoAI Web Portal on http://%s:%d", host, port)
    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
