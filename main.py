
import logging
from pathlib import Path

from nodeblocks.engine.engine_env import ENGINE_ENV
from nodeblocks.engine.logging_config import setup_logging
from nodeblocks.server.web.server import NodeServer


setup_logging(default_level=logging.DEBUG)

if __name__ == "__main__":
    root_folder = Path(__file__).parent
    ENGINE_ENV.plugin_libs_folder = root_folder / ".venv"
    
    plugins_folder = root_folder / "src"
    scenes_folder = root_folder / "scenes"

    scenes_folder.mkdir(exist_ok=True)
    server = NodeServer(
        plugins_folder=plugins_folder,
        scenes_folder=scenes_folder,
    )
    server.run()
