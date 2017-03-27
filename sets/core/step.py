import os
from sets import utility


class Step:
    """
    A cached step for processing datasets. Base class for parsing and altering
    datasets.
    """

    @classmethod
    def disk_cache(cls, basename, function, *args, method=True, **kwargs):
        """
        Cache the return value in the correct cache directory. Set 'method' to
        false for static methods.
        """
        @utility.disk_cache(basename, cls.directory(), method=method)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper(*args, **kwargs)

    @classmethod
    def download(cls, url, filename=None):
        """
        Download a file into the correct cache directory.
        """
        return utility.download(url, cls.directory(), filename)

    @classmethod
    def directory(cls, prefix=None):
        """
        Path that should be used for caching. Different for all subclasses.
        """
        prefix = prefix or utility.read_config().directory
        name = cls.__name__.lower()
        directory = os.path.expanduser(os.path.join(prefix, name))
        utility.ensure_directory(directory)
        return directory
