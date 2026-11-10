from nodeblocks.engine.plugins import PluginManifest
from nodeblocks.engine.plugins.api import Plugin


class YtorchpluginPlugin(Plugin):
    manifest = PluginManifest(
        package_id="ytorchplugin",
        plugin_version="0.1.0",
        description="This plugin implements some PyTorch nodes.",
        authors=["yDewolf"],
        
        engine_version=f">=0.2.0",
        plugin_api_version=f">=1.3.0",
        
        dependencies={"core": ">=0.2.0"},
        python_dependencies={
            "torch": ">=2.11.0",
            "torchvision": ">=0.26.0",
            "torchinfo": ">=1.8.0",
            "matplotlib": ">=3.11.2"
        }
    )
