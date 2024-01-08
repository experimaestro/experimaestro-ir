from experimaestro.launcherfinder import find_launcher

from xpmir.papers import configuration, attrs_cached_property


@configuration()
class LauncherSpecification:
    """Launcher specification

    This allows requesting computational resources such as 2 GPUs with more than
    12Go of memory)
    """

    requirements: str

    @attrs_cached_property
    def launcher(self):
        return find_launcher(self.requirements)
