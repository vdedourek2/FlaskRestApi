from flask import Blueprint, request, jsonify
from init import qa, auth  # Import initialized components from init.py

srv_par_blueprint = Blueprint("srv_par", __name__)


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
@srv_par_blueprint.route('/get_srv_par', methods=['GET'])
@auth.login_required
def process_get_srv_par():
    try:
        return jsonify(qa.get_cls_par()), 200

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
@srv_par_blueprint.route('/set_srv_par', methods=['POST'])
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