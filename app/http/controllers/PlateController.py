"""A PlateController Module."""

from masonite.controllers import Controller
from masonite.request import Request

from app.providers.DLModelProvider import DLModelProvider


class PlateController(Controller):
    """PlateController Controller Class."""

    def __init__(self, request: Request):
        """PlateController Initializer

        Arguments:
            request {masonite.request.Request} -- The Masonite Request class.
        """
        self.request = request

    def store(self, request: Request, ai: DLModelProvider):
        return {"hello": ai.count()}
