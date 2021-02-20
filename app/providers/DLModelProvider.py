"""A DLProvider Service Provider."""

import tensorflow as tf
from masonite.provider import ServiceProvider


class DLModelProvider(ServiceProvider):
    """Provides Services To The Service Container."""

    wsgi = True
    __detect_model = None
    __recognition_model = None

    def register(self):
        self.app.bind('ai', self)
        __detect_model = tf.keras.models.load_model('zc.h5')
        __recognition_model = tf.keras.models.load_model('plate.h5')

    def boot(self):
        """Boots services required by the container."""
        pass

    def model(self):
        print('model')
