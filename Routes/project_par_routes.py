# Returning 404 (Not Found) for a non-existent resource
# Returning 400 (Bad Request) for invalid request parameters
# Returning 401 (Unauthorized) for an unauthorized request
# Returning 500 (Internal Server Error) for server-side errors

from flask import Blueprint, request, jsonify
from langchain.chains.query_constructor.base import AttributeInfo
from init import qa, auth  # Import initialized components from init.py

project_par_blueprint = Blueprint("project_par", __name__)

"""
Get project parameters in dictionary format
    Parameters:
        project - project name. Is mandatory.
"""
def project_par(project):
    dictionary = qa.get_project_par(project)

    # transformation object list to dictionary list
    if "self_metadata" in dictionary:
        self_metadata = dictionary["self_metadata"]        
        self_metadata_tr = [vars(item) for item in self_metadata]
        dictionary["self_metadata"] = self_metadata_tr

    return dictionary





"""
/api/v1/projects/<project> - Get project parameters
        GET method.

    Output:
        {
            "system_msg":               system_msg,
            "api_model":                api_model,
            "add_answer_time":          add_answer_time,
            "add_citation:              add_citation,
            "citation_field":           citation_field,
            "self_doc_descr":           self_doc_descr,
            "self_metadata": [
                {
                    "description": "Field description",
                    "name": "Field name",
                    "type": "Datatype name"
                }, ...
            ],            
            "metadata_parent_field" :   metadata_parent_field,
            "routing_field":            routing_field,
            "k":                        k,
            "retriever_weights":        retriever_weights
        } 
                
    Parameters:      
        project - project name. Is mandatory.

        system_msg - partial text which will be added at the begin of the system message 
        api_model - model of the ChatGPT API.
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name gpt35, gpt4, gpt35_1106        
        add_answer_time - True - answer is with elapsed time,  False - answer is without elapsed time 
        add_citation - True - at the end of answer add web page references, False - without web page references
        citation_field - field in metadata, which is used for citation
        self_doc_descr - document description for Self Retriever
        self_metadata - list of a metadata attribute descriptions for Self Retriever
        metadata_parent_field - Metadata field for parent doc
        routing_field - field in metadata, which is used as condition  (field == value) for getting context for RAG
        k - number of chunks retrieved from a vector database
        retriever_weights - vector of initialized weights (embedding retriever, SelfQueryRetriever, BM25, MultiQueryRetriever, ParentRetriever + SelfQueryRetriever)
        
"""
@project_par_blueprint.route('/api/v1/projects/<project>', methods=['GET'])
@auth.login_required
def process_get_project_par(project):
    try:
        dictionary = project_par(project)
        
        if "error" in dictionary:
            return jsonify(dictionary), 500

        return jsonify(dictionary), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



"""
/api/v1/projects/<project>/basic - Set project parameters
        POST method.
        {
            ["system_msg":              system_msg,]
            ["api_model":               api_model,]
            ["answer_time":             answer_time,]
            ["citation:                 citation,]
            ["citation_field":          citation_field,]
            ["metadata_parent_field" :  metadata_parent_field,]
            ["routing_field":           routing_field,]
            ["k":                       k,]            
            ["erase_history":           erase_history]
        } 
           
        Parameters: (when is empty or None then are unchanged)
        project - project name (is collection name in vector db). Is mandatory.

        system_msg - system message (if is empty then is unchanged)  
        api_model - model of the ChatGPT API. (if empty then environment variable "OPENAI_API_MODEL_GPT" is used)
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time (if is None or isn't presented then is unchanged)
        citation - True - at the end of answer add web page references, False - without web page references (if is None or isn't presented then is unchanged)
        citation_field - field in metadata, which is used for citation (if is None or isn't presented then is unchanged)
        metadata_parent_field - Metadata field for parent doc (if is empty then is unchanged)  
        routing_field - field in metadata, which is used as condition  (field == value) for getting context for RAG (if is empty then is unchanged)  
        k - number of chunks retrieved from a vector database (if is None or isn't presented then is unchanged)          
        erase_history - True - question/answer history will be erased, False - question/answer history will not be erased (default False)

       Returns:
            project parameters JSON        
"""
@project_par_blueprint.route('/api/v1/projects/<project>/basic', methods=['POST'])
@auth.login_required
def process_set_project_par(project):
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:

        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        system_msg              = "" if "system_msg" not in input_data else input_data["system_msg"]
        api_model               = "" if "api_model" not in input_data else input_data["api_model"]
        answer_time             = None if "answer_time" not in input_data else input_data["answer_time"]
        citation                = None if "citation" not in input_data else input_data["citation"]
        citation_field          = None if "citation_field" not in input_data else input_data["citation_field"]
        metadata_parent_field   = None if "metadata_parent_field" not in input_data else input_data["metadata_parent_field"]
        routing_field           = None if "routing_field" not in input_data else input_data["routing_field"]
        k                       = None if "k" not in input_data else input_data["k"]
        
        erase_history           = False if "erase_history" not in input_data else input_data["erase_history"]

        qa.set_project_par(
            project = project,
            system_msg = system_msg,
            api_model = api_model,
            answer_time = answer_time,
            citation = citation,
            citation_field = citation_field,
            metadata_parent_field = metadata_parent_field,
            routing_field = routing_field,
            k = k,            
            erase_history = erase_history,
            )

        dictionary = project_par(project)
        if "error" in dictionary:
            return jsonify(dictionary), 500

        return jsonify(dictionary), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""
/api/v1/projects/<project>/weights - Setting project weights for ensemble retriever
        POST method.
        {
            "retriever_weights":    []
        } 
           
        Parameters: 
        project - project name (is collection name in vector db). Is mandatory.
        retriever_weights - [w1, w2, w3, w4, w5]
            (EmbeddingRetriever, SelfRetriever, BM25, MultiRetriever, SelfRetriever + ParentRetriever)
            wi is <0, 1> where w1 + w2 + w3 + w4 + w5 = 1
            
       Returns:
            project parameters JSON            
"""
@project_par_blueprint.route('/api/v1/projects/<project>/weights', methods=['POST'])
@auth.login_required
def process_set_retriever_par(project):
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:

        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        retriever_weights = tuple(input_data["retriever_weights"])
 
        qa.set_project_retriever(
            project = project,
            retriever_weights = retriever_weights,
            )

        dictionary = project_par(project)
        if "error" in dictionary:
            return jsonify(dictionary), 500

        return jsonify(dictionary), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
"""
/api/v1/projects/<project>/self - Seting SelfRetriever attributes
        POST method.
        {
            "project":              project,
             ["self_doc_descr":          self_doc_descr,]
            "self_metadata": [
                {
                    "name": "Field name",
                    "description": "Field description",
                    "type": "Datatype name"
                }, ...
            ],                  
        } 
           
        Parameters: 
        project - project name (is collection name in vector db). Is mandatory.
        self_doc_descr - document description for Self Retriever (if is empty then is unchanged)  
        self_metadata - list of a metadata attribute descriptions for Self Retriever (if is empty then is unchanged)  
        
       Returns:
            project parameters JSON        
"""
@project_par_blueprint.route('/api/v1/projects/<project>/self', methods=['POST'])
@auth.login_required
def process_set_self_attributes_par(project):
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:

        # Get the JSON data from the request
        input_data = request.json

        # Validate the input data (you can add more validation if needed)
        if not isinstance(input_data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400
        
        self_doc_descr = None if "self_doc_descr" not in input_data else input_data["self_doc_descr"]
        
        self_metadata = None
        if "self_metadata" in input_data:
            self_metadata = []
            for item in input_data["self_metadata"]:
                self_metadata.append(
                    AttributeInfo(
                        name = item["name"],
                        description = item["description"],
                        type = item["type"],
                        )                    
                    )
 
        qa.set_project_par(
            project = project,
            self_doc_descr = self_doc_descr,
            self_metadata = self_metadata,        
            )

        dictionary = project_par(project)
        if "error" in dictionary:
            return jsonify(dictionary), 500

        return jsonify(dictionary), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


"""
/api/v1/projects/<project>/clone/<new_project> - Creating project clone
        POST method.
            
        Parameters:
          project – existing project name
          new_project – new unique name of the clone project
          
        Returns:
            project parameters JSON

"""
@project_par_blueprint.route('/api/v1/projects/<project>/clone/<new_project>', methods=['POST'])
@auth.login_required
def process_clone_project(project, new_project):
    if auth.current_user() != "admin":
        return jsonify({"error": "For operation is needed admin permission."}), 401

    try:
        result = qa.create_clone_project(
            project = project,
            clone_project = new_project,
             )

        if result == "OK":
            dictionary = project_par(new_project)
            if "error" in dictionary:
                return jsonify(dictionary), 500

            return jsonify(dictionary), 201            

        else:
            return jsonify({"error": result}), 409




    except Exception as e:
        return jsonify({"error": str(e)}), 500





