from nodeserver.engine.plugins.protocols.plugin_manifest import PluginManifest
from nodeserver.engine.plugins.api.plugin import Plugin


class YtorchpluginPlugin(Plugin):
    manifest = PluginManifest(
        package_id="ytorchplugin",
        plugin_version="0.1.0",
        description="Description for Ytorchplugin",
        authors=["yDewolf"],
        engine_version=f">=0.2.0", # Current Engine Version
        plugin_api_version=f">=1.2.0", # Current Plugin API Version
        dependencies={"core": ">=0.2.0"},
    )
