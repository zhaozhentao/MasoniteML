"""A PlateController Module."""

from masonite.controllers import Controller
from masonite.request import Request


class PlateController(Controller):
    """PlateController Controller Class."""

    def __init__(self, request: Request):
        """PlateController Initializer

        Arguments:
            request {masonite.request.Request} -- The Masonite Request class.
        """
        self.request = request


    def store(self):
        return {"hello": "z"}
