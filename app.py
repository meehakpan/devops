from flask import Flask
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_mail import Mail

from database.db import initialize_db
from flask_restful import Api
from resources.errors import errors
import os

app = Flask(__name__)

# ok from now on you can choose env file here
# python will autmatically set the environment variable for you
# you don't have to set it up manually
os.environ['ENV_FILE_LOCATION'] = '.env'
app.config.from_envvar('ENV_FILE_LOCATION')
mail = Mail(app)



# imports requiring app and mail
from resources.routes import initialize_routes

api = Api(app, errors=errors)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

initialize_db(app)
initialize_routes(api)