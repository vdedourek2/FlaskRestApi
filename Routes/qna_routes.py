from flask import Blueprint, request, jsonify
from init import qa, auth  # Import initialized components from init.py

qna_blueprint = Blueprint("qna", __name__)

"""
/projects/<project>/qna - Question / Answer service. It cooperates with class KBAQnA. It use embeddings in the vector database Qdrant, where are prepared project embeddings data.
        POST method.
    Input:
    {
        "question":     question,
        "user_id":      user_id,
    } 

    Output:
    {
    "answer": answer
    "error":  error
    }
    
   Parameters:
        question - question (mandatory)
        user_id - unique user id (mandatory)
        project - project name (is collection name in vector db). Mandatory.
              
        answer - answer
        error - normally it is empty. It contains a text error if there is a problem    
"""

@qna_blueprint.route('/api/v1/projects/<project>/qna', methods=['POST'])
@auth.login_required
def process_qna(project):
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

        # if "project" not in input_data:
        #     return jsonify({"error": "project is missing"}), 400

        answer = qa.answer_question(
            question = input_data["question"],
            user_id = input_data["user_id"],
            # project = input_data["project"],
            project = project,
            )

        # Process output data
        output_data = {"answer": answer, "error": ""}

        return jsonify(output_data), 200

    except Exception as e:
        return jsonify({"answer":"", "error": str(e)}), 500
