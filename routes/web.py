"""Web Routes."""

from masonite.routes import Get, Post

ROUTES = [
    Get("/", "WelcomeController@show").name("welcome"),
    Post("/api/plate", "PlateController@store").name("plate_store"),
]
