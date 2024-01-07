from flask import Blueprint, request, jsonify
from init import qa, auth  # Import initialized components from init.py

qna_blueprint = Blueprint("qna", __name__)

"""
/qna - Question / Answer service. It cooperates with class KBAQnA. Use embeddings in vector database Qdrant, where are prepared embeddings data if a project.
        POST method.
    Input:
    {
        "question":     question,
        "user_id":      user_id,
        "project":      project,
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

@qna_blueprint.route('/qna', methods=['POST'])
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

        answer = qa.answer_question(
            question = input_data["question"],
            user_id = input_data["user_id"],
            project = input_data["project"],
            )

        # Process the input data (modify this part based on your logic)
        output_data = {"answer": answer, "error": ""}

        return jsonify(output_data), 200

    except Exception as e:
        return jsonify({"answer":"", "error": str(e)}), 500
