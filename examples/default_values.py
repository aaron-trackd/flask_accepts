import datetime
from dataclasses import dataclass
from marshmallow import fields, Schema, post_load
from flask import Flask, jsonify, request
from flask_accepts import accepts, responds
from flask.json import JSONEncoder


class CustomJSONEncoder(JSONEncoder):
    """Custom encoder for handling numpy integers, which cause problems"""

    def default(self, o):  # pylint: disable=method-hidden
        if callable(o):
            return str(o())
        return super().default(o)


class CogSchema(Schema):
    cog_foo = fields.String(default="cog")
    cog_baz = fields.DateTime(default=lambda: str(datetime.datetime.utcnow()))


class WidgetSchema(Schema):
    foo = fields.String(default="test string")
    baz = fields.Integer(default=42)
    flag = fields.Bool(default=False)
    date = fields.Date(default="01-01-1900")
    dec = fields.Decimal(default=42.42)
    dct = fields.Dict(default={"key": "value"})

    cog = fields.Nested(CogSchema)


def create_app(env=None):
    from flask_restx import Api, Namespace, Resource

    app = Flask(__name__)
    api = Api(app)
    api.json_encoder = CustomJSONEncoder

    @app.route("/simple/make_a_widget", methods=["POST"])
    @accepts(dict(name="some_arg", type=str), schema=WidgetSchema)
    @responds(schema=WidgetSchema)
    def post():
        from flask import jsonify

        return request.parsed_obj

    @api.route("/restx/make_a_widget")
    class WidgetResource(Resource):
        @accepts(dict(name="some_arg", type=str), schema=WidgetSchema, api=api)
        @responds(schema=WidgetSchema, api=api)
        def post(self):
            from flask import jsonify

            return request.parsed_obj

    return app


app = create_app()
if __name__ == "__main__":
    app.run(debug=True)
