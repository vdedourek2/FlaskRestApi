"""
Library instalation:
pip install load-dotenv       # environment variable service
pip install Flask             # Flask framework
pip install Flask-HTTPAuth    # Simple extension that provides Basic and Digest HTTP authentication for Flask routes.
pip install gunicorn          # LINUX server for web application running

----------------------------------------------------------------------------

Rest API server for KBA (Knowledge Base Assistent)
Using KBAQnA Class

Rest API definition:

Query/answer API:
/qna - Question / Answer service. It cooperates with class KBAQnA. Use embeddings in vector database Qdrant, where are prepared embeddings data for the project.

Server API:
/get_srv_par - Get server parameters
/set_srv_par - Set server parameters

Project API:
/get_project_par - Get project parameters
/set_project_par - Set project parameters
/set_retriever_par - Set project retrievers
--------------------------------------------------------------------------------------------------------------

Authorization (Basic Auth):
username = admin
password = .... see os.getenv("FLASK_ADMIN_API_KEY")

resp.

username = app
password = .... see os.getenv("FLASK_APP_API_KEY")

Test request:
POST http://localhost:5000/qna
input data:
{
  "question": "Ahoj",
  "user_id": "id22",
  "project": "www.multima.cz"
}

"""
import os
from flask import Flask
from werkzeug.security import check_password_hash

# Initialization
from init import users, auth  # Import initialized components from init.py

# API definition
from Routes.qna_routes import qna_blueprint                 # Query/answer API
from Routes.srv_par_routes import srv_par_blueprint         # Server API
from Routes.project_par_routes import project_par_blueprint # Project API
from Routes.diagnostic_routes import diagnostic_blueprint   # Diagnostic API


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

app.register_blueprint(qna_blueprint)
app.register_blueprint(srv_par_blueprint)
app.register_blueprint(project_par_blueprint)
app.register_blueprint(diagnostic_blueprint)


@auth.verify_password
def verify_password(username, password):
    if username in users:
        if check_password_hash(users.get(username), password):
            return username



# RUN APPLICATION -------------------------------------------------------------------------
if __name__ == '__main__':
    app.run()
