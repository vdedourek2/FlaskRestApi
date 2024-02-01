# Best practices
# https://medium.com/@nadinCodeHat/rest-api-naming-conventions-and-best-practices-1c4e781eb6a5

from flask import Blueprint, request, jsonify
from init import qa, auth  # Import initialized components from init.py
from datetime import datetime

srv_par_blueprint = Blueprint("srv_par", __name__)


"""
/api/v1/server - Get server parameters
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
@srv_par_blueprint.route('/api/v1/server', methods=['GET'])
@auth.login_required
def process_get_srv_par():
    try:
        return jsonify(qa.get_cls_par()), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


"""
/api/v1/server - Set server parameters
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
@srv_par_blueprint.route('/api/v1/server', methods=['POST'])
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

        return jsonify(qa.get_cls_par()), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
"""
/api/v1/projects - Get server projects
        GET method.

   Output:
        [project1, project2, ...] 
"""        
@srv_par_blueprint.route('/api/v1/projects', methods=['GET'])
@auth.login_required
def process_get_projects():
    try:
        return jsonify(qa.get_projects()), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
"""
/api/v1/users - Get user's history on projects
        GET method.

   Output:
        [("project":project, "user_id":user_id, "last_time":last_time, "history":[(question, answer),...]), ...]
"""        
@srv_par_blueprint.route('/api/v1/users', methods=['GET'])
@auth.login_required
def process_get_users():
    try:
        history = qa.get_users()
        
        # transform ("last_time":last_time) to JSON format
        for record in history:
            if "last_time" in record:
                # Convert the timestamp to a datetime object
                datetime_object = datetime.fromtimestamp(record["last_time"])

                # Convert the datetime object to a string in a specific format
                record["last_time"] = datetime_object.strftime("%Y-%m-%d %H:%M:%S")  

        return jsonify(history), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500