from flask import Blueprint, request, jsonify

from init import qa, auth  # Import initialized components from init.py

diagnostic_blueprint = Blueprint("diagnostic", __name__)


"""
/api/v1/projects/<project>/condensq - Getting condensed question based on user communication history.
        POST method.
 
      Input:
    {
        "question":     question,
        "user_id":      user_id
     } 

    Output:
    {
    "condensed_question": condensed question
    "history":  communication history [[question, answer], …]
    "error":  error
    }
    
   Parameters:
        question - question (mandatory)
        user_id - unique user id (mandatory)
        project - project name. Mandatory.
              
        condensed_question – condensed question created on the communication history
        error - normally it is empty. It contains a text error if there is a problem
"""
@diagnostic_blueprint.route('/api/v1/projects/<project>/condensq', methods=['POST'])
@auth.login_required
def process_get_condensed_question(project):
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:
        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        if "question" not in input_data:
            return jsonify({"error": "Missing question parameter"}), 400

        if "user_id" not in input_data:
            return jsonify({"error": "Missing user_id parameter"}), 400

        result = qa.get_condensed_question(
            project = project,
            question = input_data["question"],
            user_id = input_data["user_id"],
             )

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
"""
/api/v1/projects/<project>/selfcondition - Getting generated condition for SelfRetriever
        POST method.
 
 Input:
    {
        "question":     question
    } 

    Output:
    {
    "condition": condition for SelfRetriever
    }
    
   Parameters:
        question - question (mandatory)
        project - project name. Mandatory.
              
        condition – generated condition for SelfRetriever

"""
@diagnostic_blueprint.route('/api/v1/projects/<project>/selfcondition', methods=['POST'])
@auth.login_required
def process_get_self_condition(project):
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:
        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        if "question" not in input_data:
            return jsonify({"error": "Missing question parameter"}), 400

        self_condition = qa.get_self_condition(
            project = project,
            question = input_data["question"],
             )
        
        self_condition_str = f"{self_condition}"
        result = {"condition": self_condition_str}
 
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""
/api/v1/projects/<project>/context - Getting generated context for retriever
        POST method.
 
 Input:
    {
        "question":     question
     } 

 Output:
    {
    "context": [chunk list of the context from vector database]
    "error":  error
    }
    
   Parameters:
        question - question (mandatory)
        project - project name. Mandatory.
              
        context – chunk list of the context from vector database
        error - normally it is empty. It contains a text error if there is a problem


"""
@diagnostic_blueprint.route('/api/v1/projects/<project>/context', methods=['POST'])
@auth.login_required
def process_get_context(project):
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:
        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        if "question" not in input_data:
            return jsonify({"error": "Missing question parameter"}), 400

        context_list = qa.get_context(
            project = project,
            question = input_data["question"],
             )
        
        context = [vars(item) for item in context_list]
        
        result = {"context": context}
 
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500