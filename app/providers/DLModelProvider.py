"""A DLProvider Service Provider."""

from masonite.provider import ServiceProvider


class DLModelProvider(ServiceProvider):
    """Provides Services To The Service Container."""

    wsgi = True
    __counter = 0

    def register(self):
        self.app.bind('ai', self)

    def boot(self):
        """Boots services required by the container."""
        pass

    def count(self):
        self.__counter += 1
        return self.__counter
