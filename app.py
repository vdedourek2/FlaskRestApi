"""
Library instalation:
pip install load-dotenv       # environment variable service
pip install Flask             # Flask framework
pip install Flask-HTTPAuth    # Simple extension that provides Basic and Digest HTTP authentication for Flask routes.
pip install gunicorn          # LINUX server for web application running
"""



"""
Rest API server for KBA (Knowledge Base Assistent)
Using KBAQnA Class

Rest API definition:

/qna - Question / Answer service. It cooperates with class KBAQnA. Use embeddings in vector database Qdrant, where are prepared embeddings data if a project.
        POST method.
    Input:
    {
        "question":     question,
        "user_id":      user_id,
        "project":      project,
        [system_msg":   system_msg,]
        ["api_type":     api_type,]
        ["api_base":     api_base,]
        ["api_key":      api_key,]
        ["api_version":  api_version,]
        ["api_model":    api_model,]
    } 

    Output:
    {
    "answer": answer
    "error":  error
    }
    
    Parameters:
        question - question (is mandatory)
        user_id - unique user id (is mandatory)
        project - project name (is collection name in vector db). Is mandatory.
        system_msg - partial text which will be added at the begin of the system message (can be empty)
        api_type - OpenAI type - open_ai, azure (if empty then environment variable "OPENAI_API_TYPE" is used )
        api_base - URL base of the ChatGPT API (if empty then environment variable "OPENAI_API_BASE" is used 
        api_key - API key of the ChatGPT (if empty then environment variable "OPENAI_API_KEY" is used)
        api_version - version of the ChatGPT API (if empty then environment variable "OPENAI_API_VERSION" is used)
        api_model - model of the ChatGPT API. (if empty then environment variable "OPENAI_API_MODEL_GPT" is used)
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name
            
        answer - answer
        error - normally it is empty. It contains a text error if there is a problem
--------------------------------------------------------------------------------------------------------------
/get_srv_par - Get server parameters
        GET method.

   Output:
        {
            "db_type":              db_type,
            "db_dir":               db_dir,
            "system_msg":           system_msg,
            "k_history":            k_history,
            "time_limit_history":   time_limit_history,
            "verbose":              verbose,
            "answer_time":          answer_time,
        } 
        
    Parameters:
        db_type - vector database type  (if empty then unchanged): 
            local - local Chroma DB in db directory, 
            qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
        db_dir - directory, where is saved local vector Chroma db (only for db = local)
        system_msg - partial text which will be added at the begin of the system message (can be empty)
        k_history - the maximum length of history that is used for the conversation
        time_limit_history - the time interval in seconds after which the history is erased
        verbose - True - logging process question/answer to system output, False - without logging
        answer_time - True - the answer contains the time spent in seconds,  False - answer is without spent time        
        
--------------------------------------------------------------------------------------------------------------

/set_srv_par - Set server parameters
        POST method.
       {
            ["db_type":              db_type,]
            ["db_dir":               db_dir,]
            ["system_msg":           system_msg,]
            ["k_history":            k_history,]
            ["time_limit_history":   time_limit_history,]
            ["verbose":              verbose,]
            ["answer_time":          answer_time,]
            ["erase_history":        erase_history]
            
        }         

    Parameters(if parameter isn't used then is unchanged):
        db_type - vector database type  (if empty then unchanged): 
            local - local Chroma DB in db directory, 
            qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
        db_dir - directory, where is saved local vector Chroma db (only for db = local)
        system_msg - partial text which will be added at the begin of the system message ((if empty then it is unchanged))
        k_history - the maximum length of history that is used for the conversation
        time_limit_history - the time interval in seconds after which the history is erased
        verbose - True - logging process question/answer to system output, False - without logging
        answer_time - True - the answer contains the time spent in seconds,  False - answer is without spent time
        
    Others:
        erase_history - True - question/answer history will be erased, False - question/answer history will not be erased
        Default False.
        
--------------------------------------------------------------------------------------------------------------
        
/get_project_par - Get project parameters
        GET method.

    Output:
        {
            "project":              project,
            "system_msg":           system_msg,
            "api_model":            api_model,
            "answer_time":          answer_time,
        } 
                
    Parameters:      
        project - project name. Is mandatory.

        Parameters:
        system_msg - partial text which will be added at the begin of the system message 
        api_model - model of the ChatGPT API.
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time 

--------------------------------------------------------------------------------------------------------------

/set_project_par - Set project parameters
        POST method.
        {
            "project":              project,
            ["system_msg":           system_msg,]
            ["api_model":            api_model,]
            ["answer_time":          answer_time,]
            ["erase_history":        erase_history]
        } 
           
        Parameters: (when is empty or None then are unchanged)
        project - project name (is collection name in vector db). Is mandatory.

        system_msg - partial text which will be added at the begin of the system message (if is empty then is unchanged)  
        api_model - model of the ChatGPT API. (if empty then environment variable "OPENAI_API_MODEL_GPT" is used)
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time (if is None or isn't presented then is unchanged)

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
from dotenv import load_dotenv      # python-dotenv
import os
from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth # Flask-HTTPAuth
from werkzeug.security import generate_password_hash, check_password_hash
from Processing.qna_lch_mod import KBAQnA


system_msg = "You are AI Assistant named Vanda."
qa = KBAQnA(
    db_type = "qdrant",
    db_dir = "db",  
    system_msg = system_msg,
    answer_time = False,
    verbose = False)

app = Flask(__name__)

auth = HTTPBasicAuth()

load_dotenv()

# application API Key definition
users = {
    "admin": generate_password_hash(os.getenv("FLASK_ADMIN_API_KEY")),  # administrator API key
    "app":   generate_password_hash(os.getenv("FLASK_APP_API_KEY")),    # application API key
}

app.secret_key = os.getenv("FLASK_SECRET_KEY")
print("Start KBA")


@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username


"""
/qna - Question / Answer service. It cooperates with class KBAQnA. Use embeddings in vector database Qdrant, where are prepared embeddings data if a project.
        POST method.
    Input:
    {
        "question":     question,
        "user_id":      user_id,
        "project":      project,
        [system_msg":   system_msg,]
        ["api_type":     api_type,]
        ["api_base":     api_base,]
        ["api_key":      api_key,]
        ["api_version":  api_version,]
        ["api_model":    api_model,]
    } 

    Output:
    {
    "answer": answer
    "error":  error
    }
"""
@app.route('/qna', methods=['POST'])
@auth.login_required
def process_qna():
    try:

        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400

        if "question" not in input_data:
            return jsonify({"error": "question is missing"}), 400

        if "user_id" not in input_data:
            return jsonify({"error": "user_id is missing"}), 400

        if "project" not in input_data:
            return jsonify({"error": "project is missing"}), 400

        system_msg  = "" if "system_msg" not in input_data else input_data["system_msg"]
        api_type    = "" if "api_type" not in input_data else input_data["api_type"]
        api_base    = "" if "api_base" not in input_data else input_data["api_base"]
        api_key     = "" if "api_key" not in input_data else input_data["api_key"]
        api_version = "" if "api_version" not in input_data else input_data["api_version"]
        api_model   = "" if "api_model" not in input_data else input_data["api_model"]

        answer = qa.answer_question(
            question = input_data["question"],
            user_id = input_data["user_id"],
            project = input_data["project"],
            system_msg = system_msg,
            api_type = api_type,
            api_base = api_base,
            api_key = api_key,
            api_version = api_version,
            api_model = api_model,
            )

        # Process the input data (modify this part based on your logic)
        output_data = {"answer": answer, "error": ""}

        return jsonify(output_data), 200

    except Exception as e:
        return jsonify({"answer":"", "error": str(e)}), 500


"""
/get_srv_par - Get server parameters
        GET method.

   Output:
        {
            "db_type":              db_type,
            "db_dir":               db_dir,
            "system_msg":           system_msg,
            "k_history":            k_history,
            "time_limit_history":   time_limit_history,
            "verbose":              verbose,
            "answer_time":          answer_time,
        } 
        
    Parameters:
        db_type - vector database type  (if empty then unchanged): 
            local - local Chroma DB in db directory, 
            qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
        db_dir - directory, where is saved local vector Chroma db (only for db = local)
        system_msg - partial text which will be added at the begin of the system message (can be empty)
        k_history - the maximum length of history that is used for the conversation
        time_limit_history - the time interval in seconds after which the history is erased
        verbose - True - logging process question/answer to system output, False - without logging
        answer_time - True - the answer contains the time spent in seconds,  False - answer is without spent time  
"""        
@app.route('/get_srv_par', methods=['GET'])
@auth.login_required
def process_get_srv_par():
    try:
        return jsonify(qa.get_cls_par()), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



"""
/get_project_par - Get project parameters
        GET method.

    Output:
        {
            "project":              project,
            "system_msg":           system_msg,
            "api_model":            api_model,
            "answer_time":          answer_time,
        } 
                
    Parameters:      
        project - project name. Is mandatory.

        Parameters:
        system_msg - partial text which will be added at the begin of the system message 
        api_model - model of the ChatGPT API.
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time 
"""
@app.route('/get_project_par/<project>', methods=['GET'])
@auth.login_required
def process_get_project_par(project):
    try:
        return jsonify(qa.get_project_par(project)), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



"""
/set_srv_par - Set server parameters
        POST method.
       {
            ["db_type":              db_type,]
            ["db_dir":               db_dir,]
            ["system_msg":           system_msg,]
            ["k_history":            k_history,]
            ["time_limit_history":   time_limit_history,]
            ["verbose":              verbose,]
            ["answer_time":          answer_time,]
            ["erase_history":        erase_history]
        }         

    Parameters(if parameter isn't used then is unchanged):
        db_type - vector database type  (if empty then unchanged): 
            local - local Chroma DB in db directory, 
            qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
        db_dir - directory, where is saved local vector Chroma db (only for db = local)
        system_msg - partial text which will be added at the begin of the system message ((if empty then it is unchanged))
        k_history - the maximum length of history that is used for the conversation
        time_limit_history - the time interval in seconds after which the history is erased
        verbose - True - logging process question/answer to system output, False - without logging
        answer_time - True - the answer contains the time spent in seconds,  False - answer is without spent time
        
    Others:
        erase_history - True - question/answer history will be erased, False - question/answer history will not be erased
        Default False.
"""        
@app.route('/set_srv_par', methods=['POST'])
@auth.login_required
def process_set_srv_par():
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:

        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400
 
        db_type             = "" if "db_type" not in input_data else input_data["db_type"]
        db_dir              = "" if "db_dir" not in input_data else input_data["db_dir"]
        system_msg          = "" if "system_msg" not in input_data else input_data["system_msg"]
        k_history           = None if "k_history" not in input_data else input_data["k_history"]
        time_limit_history  = None if "time_limit_history" not in input_data else input_data["time_limit_history"]
        verbose             = None if "verbose" not in input_data else input_data["verbose"]
        answer_time         = None if "answer_time" not in input_data else input_data["answer_time"]
        erase_history       = False if "erase_history" not in input_data else input_data["erase_history"]

        qa.set_cls_par(
            db_type = db_type,
            db_dir = db_dir,
            system_msg = system_msg,
            k_history = k_history,
            time_limit_history = time_limit_history,
            verbose = verbose,
            answer_time = answer_time,
            erase_history = erase_history,
            )

        return jsonify({}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""
/set_project_par - Set project parameters
        POST method.
        {
            "project":              project,
            ["system_msg":           system_msg,]
            ["api_model":            api_model,]
            ["answer_time":          answer_time,]
            ["erase_history":        erase_history]
        } 
           
        Parameters: (when is empty or None then are unchanged)
        project - project name (is collection name in vector db). Is mandatory.

        system_msg - partial text which will be added at the begin of the system message (if is empty then is unchanged)  
        api_model - model of the ChatGPT API. (if empty then environment variable "OPENAI_API_MODEL_GPT" is used)
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time (if is None or isn't presented then is unchanged)
"""
@app.route('/set_project_par', methods=['POST'])
@auth.login_required
def process_set_project_par():
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:

        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        if "project" not in input_data:
            return jsonify({"error": "project is missing"}), 400
 
        system_msg          = "" if "system_msg" not in input_data else input_data["system_msg"]
        api_model           = "" if "api_model" not in input_data else input_data["api_model"]
        answer_time         = None if "answer_time" not in input_data else input_data["answer_time"]
        erase_history       = False if "erase_history" not in input_data else input_data["erase_history"]

        qa.set_project_par(
            project = input_data["project"],
            system_msg = system_msg,
            api_model = api_model,
            answer_time = answer_time,
            erase_history = erase_history,
            )

        return jsonify({}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run()
