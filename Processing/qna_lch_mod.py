''' 
KBAQnA - Class for talking with Knowledge Base Assistent on Vectore database data
        which are created by KBAIndex class (based on Langchain)

Sources:
https://www.linkedin.com/pulse/build-qa-bot-over-private-data-openai-langchain-leo-wang/
https://betterprogramming.pub/build-a-chatbot-for-your-notion-documents-using-langchain-and-openai-e0ad7b903b84

Library instalation:
pip install load-dotenv       # environment variable service
pip install langchain         # framework for LLM
pip install langchain-openai  # framework for openai functions
pip install openai            # OpenAI
pip install chromadb          # Chromadb database API
pip install qdrant-client     # Qdrant database API
pip install pgvector          # pgvector extension for PostgreSQL
pip install lark              # Needed for SelfQuerying
pip install funcy             # funcy library
'''

from dotenv import load_dotenv      # python-dotenv
import os
import time
import ast
# import pickle
import logging
from dataclasses import replace
from funcy import print_durations, project
from openai import OpenAI, AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
# from langchain_community.utils.openai_functions import (
#     convert_pydantic_to_openai_function,
# )

from operator import itemgetter
from qdrant_client import QdrantClient
import chromadb
from Processing.db_mod import KBADatabase
from Processing.project_mod import Project
from Processing.pgvector_client_mod import PgvectorClient

class KBAQnA(object):
    '''
    ### Class for talking with Knowledge Assistent on Vectore database data,
        which are created by KBAIndex class (based on Langchain)
    -----------------------------------------------------------------------
    db_type - Select option: 
        local - local Chroma DB in db directory, 
        qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
        pgvector - pgvector extension in PostgreSQL. Needs environment variables: SQLDB_HOST, SQLDB_DATABASE, SQLDB_UID, SQLDB_PWD        
    db_dir - directory, where is saved local Chroma db (only for db = local)
    
    system_msg - partial text which will be added at the begin of the Chat GPT system message (short chatbot description)
    k - number of chunks retrieved from a vector database
    k_history - the maximum length of history that is used for the conversation
    time_limit_history - the time interval in seconds after which the history is cleared
    verbose - True - logging process question/answer, False - without logging
    answer_time - True - answer is with elapsed time,  False - answer is without elapsed time
    https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/
    https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query
    https://python.langchain.com/docs/integrations/retrievers/self_query/qdrant_self_query
    '''
    def __init__(self,
        db_type:str = "local",
        db_dir:str = "",
        system_msg:str  = "",
        k_history:    int  = 3,
        time_limit_history: int = 3600,
        verbose:      bool = False,
        answer_time:  bool = False,
        ):
 
        load_dotenv()
        
        self.db_type            = db_type
        self.db_dir             = db_dir
        self.k_history          = k_history
        self.verbose            = verbose
        self.answer_time        = answer_time
        self.time_limit_history = time_limit_history
        self.system_msg         = system_msg

        api_type = os.getenv("OPENAI_API_TYPE")
        api_base = os.getenv("OPENAI_API_BASE") if api_type == "open_ai" else os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("OPENAI_API_KEY") if api_type == "open_ai" else os.getenv("AZURE_OPENAI_API_KEY")

        # setting embeddings
        if (api_type == "open_ai"):
            self.embeddings = OpenAIEmbeddings(
                base_url = api_base,
                api_key=api_key,
                )
        else:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("OPENAI_API_MODEL_ADA"),
                azure_endpoint=api_base,
                )
         
        # LLM for condensed questions
        api_model_gpt4 = os.getenv("OPENAI_API_MODEL_GPT4")

        if (os.getenv("OPENAI_API_TYPE") == "open_ai"):
            self.llm_condensed = ChatOpenAI(
                temperature=0,
                model_name=api_model_gpt4,
                )
        else:
            self.llm_condensed = AzureChatOpenAI(
                temperature=0,
                deployment_name=api_model_gpt4,
                )
             
         
        # vector database connection
        if db_type == "local":
            self.db_client = chromadb.PersistentClient(path=db_dir)

        if db_type == "qdrant":
            self.db_client = QdrantClient(
                url = os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                prefer_grpc=True,               
                ) 
            
        if db_type == "pgvector":
            self.db_client = PgvectorClient()
            
        # SQL database connection
        self.db = KBADatabase()
        
        # setup logging for MultiQueryRetriever
        if verbose:
            logging.basicConfig()
            logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
        
                
        self.projects = {}  # projects parameters dictionary {"project_name": Project class }



        
    def set_cls_par(self,
        db_type:      str = "",
        db_dir:       str = "",
        system_msg:   str  = "",
        k_history:    int  = None,
        time_limit_history: int = None,
        verbose:      bool = None,
        answer_time:  bool = None,
        erase_history:bool = False,
        ):
        '''
        Set class parameters.
        Vector database is reconnected.
        SQL database is reconnected.
        Conversation history can be erased.
        ----------------------------------------------------------------------------------------
        Parameters: (when is empty or None then are unchanged)
        db_type - vector database type  (if empty then unchanged): 
            local - local Chroma DB in db directory, 
            qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
        db_dir - directory, where is saved local vector Chroma db (only for db = local) (if empty then unchanged)
        system_msg - short chatbot description (if empty then it is unchanged)
        k_history - the maximum length of history that is used for the conversation (if None then unchanged)
        time_limit_history - the time interval in seconds after which the history is cleared (if None then unchanged)
        verbose - True - logging process question/answer, False - without logging (if None then unchanged)
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time (if None then unchanged)

        Others:
        erase_history - True - question/answer history will be erased, False - question/answer history will not be erased 
        '''       
        projects = self.projects
        
        if db_type == "":
            db_type = self.db_type

        if db_dir == "":
            db_dir = self.db_dir

        if system_msg == "":
            system_msg = self.system_msg

        if k_history == None:
            k_history = self.k_history

        if time_limit_history == None:
            time_limit_history = self.time_limit_history

        if verbose == None:
            verbose = self.verbose

        if answer_time == None:
            answer_time = self.answer_time
 
        self.__init__(
            db_type = db_type,
            db_dir = db_dir,
            system_msg  = system_msg,
            k_history  = k_history,
            time_limit_history = time_limit_history,
            verbose = verbose,
            answer_time = answer_time,           
            )
        
        self.projects = projects
        
        if erase_history: 
            for item in self.projects:
                item.memory.clear_history()
        
        self._set_retriever("") # setting all retrievers

        # writing to protocol
        protocol = f"cls_par: DB type = {db_type}, DB dir = {db_dir}, System message = {system_msg}, K history = {k_history}, Time limit history = {time_limit_history}, Answer time = {answer_time}"
        self.db.write_db_protocol(project = "",  protocol =protocol,  )



    def get_cls_par(self) -> dict:
        '''
        Get class parameters.
        ----------------------------------------------------------------------------------------
        return parameters in Dictionary format:
        db_type - vector database type  (if empty then unchanged): 
            local - local Chroma DB in db directory, 
            qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
        db_dir - directory, where is saved local vector Chroma db (only for db = local)
        system_msg - short chatbot description
        k_history - the maximum length of history that is used for the conversation
        time_limit_history - the time interval in seconds after which the history is cleared
        verbose - True - logging process question/answer, False - without logging
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time
        '''       
        return {
            "db_type":              self.db_type,
            "db_dir":               self.db_dir,
            "system_msg":           self.system_msg,
            "k_history":            self.k_history,
            "time_limit_history":   self.time_limit_history,
            "verbose":              self.verbose,
            "answer_time":          self.answer_time,
            }        

    def set_project_par(self,
        project:str="",
        system_msg:str  = "",
        api_model:str = "",
        answer_time:  bool = None,
        citation:     bool = None,
        citation_field:str = None,
        self_doc_descr: str = None,
        self_metadata:list = None,
        metadata_parent_field:str = None,        
        k:int = None,
        routing_field:str = None,       # field in metadata, which is used as condition  (field = value) for getting context for RAG
        routing_text:str = None,        # text which is used for routing to RAG model
        
        erase_history:bool = False,
        ):
        '''
        Set project parameters.
        Conversation project history can be erased.
        ----------------------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.

        return parameters in Dictionary format:
        system_msg - partial text which will be added at the begin of the system message (if is empty then is unchanged)  
        api_model - model of the ChatGPT API. (if empty then environment variable "OPENAI_API_MODEL_GPT" is used)
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name gpt4, gpr35
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time (if is None then is unchanged)
        citation - True - at the end of answer add web page references, False - without web page references (if is None then is unchanged)
        citation_field - field in metadata, which is used for citation (if is None then is unchanged)
        self_doc_descr - document description for Self Retriever (if is None then is unchanged)
        self_metadata - metadata description list for Self Retriever (if isn't empty then is used Self Retriever)
        metadata_parent_field - Metadata field for parent doc (if is None then is unchanged)
        k - number of chunks retrieved from a vector database (if None then unchanged)
        routing_field - field in metadata, which is used as condition  (field == value) for getting context for RAG (if is None then is unchanged)
        erase_history - True - erase history for the project, False - history isn't erased
        '''       
        
        self._get_project_par(project)
        
        item2 = self.projects[project]

        # check correct model (only for OpenAi)
        if api_model != "":
            if ( os.getenv("OPENAI_API_TYPE") == "open_ai"):
                client = OpenAI()
                if api_model in [item.id for item in client.models.list().data]:
                    # item["api_model"] = api_model
                    item2.api_model = api_model
                else:
                    print(f"Error in set_project_par: Model '{api_model}' doesn't exist")
            else:
                item2.api_model = api_model

        if system_msg != "":
            item2.system_msg = system_msg
        
        if answer_time != None:
            item2.add_answer_time = answer_time
            
        if citation != None:
            item2.add_citation = citation

        if citation_field != None:
            item2.citation_field = citation_field

        if self_doc_descr != None:
            item2.self_doc_descr = self_doc_descr
        
        if self_metadata != None:
            item2.self_metadata = self_metadata

        if metadata_parent_field != None:
            item2.metadata_parent_field = metadata_parent_field
            
        if k != None:
            item2.k = k
            
        if routing_field != None:
            item2.routing_field = routing_field

        self.projects[project] = item2
        
        # setting llm in self.projects[project].llm
        self.projects[project].set_llm()
        
        # set retriever
        self.projects[project].set_retriever(
            verbose = self.verbose,                
            )    


        if erase_history:
            self.projects[project].memory.clear_history()   # MEM

        # writing to protocol
        protocol = f"proj_par: Model = {item2.api_model}, System message = {item2.system_msg}, Answer time = {item2.add_answer_time}, Citation = {item2.add_citation}"
        self.db.write_db_protocol(
            project = project,
            protocol =protocol,  )



    def get_project_par(self,
        project:str = "",
                        
    ) -> dict:
        '''
        Get project parameters.
        ----------------------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.

        Parameters:
        collection - collection name in a vector database
        clone - True - clone project which was created from project <collection>, False - normal project (project = collection)
        system_msg - partial text which will be added at the begin of the system message 
        api_model - model of the ChatGPT API.
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time
        citation - True - at the end of answer add web page references, False - without web page references
        citation_field - field in metadata, which is used for citation
        self_doc_descr - document description for Self Retriever
        self_metadata - metadata description list for Self Retriever        
        metadata_parent_field - Metadata field for parent doc
        routing_field - field in metadata, which is used as condition  (field == value) for getting context for RAG
        k - number of chunks retrieved from a vector database
        retriever_weights - weights of retrievers (embedding retriever, SelfQueryRetriever, BM25, MultiQueryRetriever, ParentSelfQueryRetriever)
        '''       
        
        if project not in self.projects:
            return {"error": f"Projekt {project} neexistuje"}

        item = self.projects[project]
  
        return {
            "project":                  item.project,
            "collection":               item.collection,
            "clone":                    item.clone,
            "system_msg":               item.system_msg,
            "api_model":                item.api_model,
            "add_answer_time":          item.add_answer_time,
            "add_citation":             item.add_citation,
            "citation_field":           item.citation_field,
            "self_doc_descr":           item.self_doc_descr,
            "self_metadata":            item.self_metadata,
            "metadata_parent_field" :   item.metadata_parent_field,
            "routing_field":            item.routing_field,
            "k":                        item.k,
            "retriever_weights":        item.retriever_weights, 
            
            }


    def create_clone_project(self,
        project:str = "",
        clone_project:str = "",
    )->str:
        '''
        Getting generated condition for SelfRetriever.
        ----------------------------------------------------------------------------------------
        Parameters:
            project - project name. Is mandatory.
            clone_project - new clone project (mandatory)
            
        returns:
        OK - clone project was created
        "Error: error description"
       
        '''
        result = "OK"        

        try:
            item = replace(self._get_project_par(project))
            
            # checking parameters
            if clone_project in self.projects:
                return f"Error: klonovaný projekt {clone_project} již existuje. Nelze ho znovu vytvořit."
            
            item.project = clone_project
            item.clone = True
            self.projects[clone_project] = item

        except Exception as e:  
            result = f"Error: {e}"
           
        return result




    def get_condensed_question(self,
        project:str = "",
        question:str = "",
        user_id:str = "",
    ) -> dict:
        '''
        Get project parameters.
        ----------------------------------------------------------------------------------------
        Parameters:
            project - project name. Is mandatory.
            question - question (mandatory)
            user_id - unique user id (mandatory)
              
        returns:
        {
            "condensed_question": condensed question
            "history":  communication history [[question, answer], …]
            "error":  error
        }

            condensed_question – condensed question created on the communication history
            error - normally it is empty. It contains a text error if there is a problem

        '''
        result = {"condensed_question": "", "history": None, "error": "" }        

        try:
            result["condensed_question"] = self._get_condensed_question(project, user_id, question)
            result["history"] = self.projects[project].memory.get_history(user_id)     # MEM
 
        except Exception as e:
            result["condensed_question"] = self._filter_error(e)
            result["error"] = "Azure policy error"
            
        return result
    
    def get_self_condition(self,
        project:str = "",
        question:str = "",
    ):
        '''
        Getting generated condition for SelfRetriever.
        ----------------------------------------------------------------------------------------
        Parameters:
            project - project name. Is mandatory.
            question - question (mandatory)
                
        returns:
            generated condition for SelfRetriever


        '''
        self._get_project_par(project)
           
        return self.projects[project].get_self_condition(question)

    # @print_durations() 
    def get_context(self,
        project:str = "",
        question:str = "",
    ) -> list:
        '''
        Getting generated context for retriever
        ----------------------------------------------------------------------------------------
        Parameters:
            project - project name. Is mandatory.
            question - question (mandatory)
                
        returns:
        [chunk list of the context from vector database]
   

        '''
        retriever = self._get_retriever( project=project )
        contexts = retriever.get_relevant_documents(question)
            
        return contexts



    def set_project_retriever(self,
        project:str="",
        retriever_weights:tuple  = (1, 0, 0, 0, 0),
        ):
        '''
        Set ensemble project retriever weights.
        ----------------------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.
        retriever_weights - weight vector of ensemble retriever. Weight are in interval <0, 1>
        vector of (EmbeddingRetriever, SelfRetriever, BM25, MultiRetriever, SelfRetrieverParent)
        '''       
        
        self._get_project_par(project)  # if projects par aren't setup then are initialized
 
        self.projects[project].retriever_weights = retriever_weights
         
        self._set_retriever(project)    # setting retriever

        # writing to protocol
        protocol = f"weights_par: EmbeddingRetriever = {retriever_weights[0]}, SelfRetriever = {retriever_weights[1]}, \
BM25 = {retriever_weights[2]}, MultiRetriever = {retriever_weights[3]}, SelfRetrieverParent = {retriever_weights[4]}"
        self.db.write_db_protocol(
            project = project,
            protocol =protocol,  )
        




    
    def answer_question(self,
        question:str ="Co je Keymate?",
        user_id: str ="",        # user id
        project:str = "",
     ) -> str:
        """
        Answer a question 
        -------------------------------------------------------------------------
        question - question (is mandatory)
        user_id - unique user id (is mandatory)
        project - project name (is collection name in vector db). Is mandatory.
 
        returns answer
        """
        st = time.time()
        
        # getting project parameters self.project[project]
        project_par = self._get_project_par(project)

        # nove reseni RAG
        context_num = 0
        try:
            # creating independent question from question + conversation history
            condensed_question = self._get_condensed_question(project, user_id, question)
            
            # generating routing value, which is used for precise question
            if project_par.routing_field:
                filter_value = self._get_routing_value(project = project, question = condensed_question)
                if filter_value:
                    condensed_question += f" {project_par.routing_field} = '{filter_value}'"

            # generating answer from RAG
            chain = self._get_qa_lcel (project, project_par.system_msg, condensed_question)

            # https://python.langchain.com/docs/modules/model_io/llms/token_usage_tracking        
            with get_openai_callback() as cb:
                result_chain = chain.invoke(input = condensed_question)
 
            # number chunks of the context
            context_num = len(result_chain["documents"])
 
            if context_num == 0:
                answer = "Nevím"
            else:
                answer = result_chain["answer"]
        except Exception as e:
            answer = self._filter_error(e)
            condensed_question = None
            cb = OpenAICallbackHandler()
            cb.prompt_tokens = 0
            cb.completion_tokens = 0
            cb.total_cost = 0      
        
        self.projects[project].memory.add_history(user_id, question, answer)    # save question/answer to the conversation history # MEM

        et = time.time()
  

        # write question/answer to DB log
        if question == condensed_question:
            condensed_question = None
            
        self.db.write_db_log(
            project = project,
            user_id = user_id,
            question = question,
            condensed_question = condensed_question,
            answer=answer,
            api_model = project_par.api_model,
            elapsed_time = et - st,
            prompt_tokens = cb.prompt_tokens,
            completion_tokens = cb.completion_tokens,
            total_cost = cb.total_cost,            
        )
          
        if project_par.add_citation and context_num > 0 and not answer.startswith("Nevím"):
            answer += self._get_citation_chain(result_chain, project_par.citation_field)


        if project_par.add_answer_time:
            answer += f" ({round(et - st, 3)} s)"
            
        return answer

    def get_dataset_item(self,
        question:str ="Co je Keymate?",
        project:str = "",
        ):
        """
        Generate dataset item for one question. It is used in RAGAS.
        Generating doesn't work with history
        -------------------------------------------------------------------------
        question - question (is mandatory)
        project - project name (is collection name in vector db). Is mandatory.
 
        returns data_item = {'question': question, 'answer': answer, 'contexts': [...]}  
        """
 
        # getting project parameters
        project_par = self._get_project_par(project)

          # nove reseni RAG
        context_num = 0; contexts = []
        try:
            # generating routing value, which is used for precise question
            if project_par.routing_field:
                filter_value = self._get_routing_value(project = project, question = question)
                if filter_value:
                    question += f" {project_par.routing_field} = '{filter_value}'"

            # generating answer from RAG
            chain = self._get_qa_lcel (project, project_par.system_msg, question)

            # https://python.langchain.com/docs/modules/model_io/llms/token_usage_tracking        
            result_chain = chain.invoke(input = question)
 
            # number chunks of the context
            context_num = len(result_chain["documents"])
 
            if context_num == 0:
                answer = "Nevím"
            else:
                answer = result_chain["answer"]
                contexts = result_chain["contexts"]
 
        except Exception as e:
            answer = "Nevím. " + self._filter_error(e)
 
 
        return {'question': question, 'answer': answer, 'contexts': contexts}    

   

    def _get_qa_lcel(self,
        project:str = "",
        system_msg:str = "",
        query:str="",
        
     ):
        """
        Return LCEL qa object for Question/answer
        -------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.
        system_msg - partial text which will be added at the begin of the system message
        query - query only for get context for verbose == True

        returns (chain for generatung answer)
        """

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
     
        # Retrieval-augmented generation (RAG)
        # https://python.langchain.com/docs/use_cases/question_answering/
        if system_msg == "":
            GENERAL_SYSTEM_TEMPLATE = """Jsi AI asistent a odpovídáš pouze na základě Vědomostí dodaných uživatelem. \
Pokud není na základě uvedených vědomostí možné jednoznačně odpovědět na otázku, odpověz "Nevím".

Příklad, pokud informace není obsažena ve vědomostech:
Uživatel: kde najdu vysokou školu v Lounech
Asistent: Nevím"""
        else:
            GENERAL_SYSTEM_TEMPLATE = system_msg

        GENERAL_USER_TEMPLATE = """Tvoje vědomosti:

{context}

#####
Otázka: {question} 
Odpovídej na základě výše uvedených vědomostí. \
Pokud nelze na základě vědomostí jednoznačně odpovědět, tvoje odpověď bude pouze "Nevím"."""

        messages = [
                    SystemMessagePromptTemplate.from_template(GENERAL_SYSTEM_TEMPLATE),
                    HumanMessagePromptTemplate.from_template(GENERAL_USER_TEMPLATE)
        ]
        qa_prompt = ChatPromptTemplate.from_messages( messages )

        llm = self.projects[project].llm

        retriever = self._get_retriever( project=project )
        
        if self.verbose:
          
            # print SelfQueryRetriever condition
            self_condition = self.projects[project].get_self_condition(query)
            if self_condition != None:
                print(f"*** Self condition: {query}\n{self_condition}\n")               

            # print context
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([str(doc) for doc in docs])
            print(f"*** Context:\n\n{context}\n")

        rag_chain_from_docs = (
            {
                "context": lambda input: format_docs(input["documents"]),
                "question": itemgetter("question"),
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        
        rag_chain_with_source = RunnableParallel(
            {"documents": retriever, "question": RunnablePassthrough()}
        ) | {
            "documents": lambda input: [doc.metadata for doc in input["documents"]],
            "answer": rag_chain_from_docs,
            "contexts": lambda input: [doc.page_content for doc in input["documents"]],
        }

        return rag_chain_with_source

    # @print_durations() 
    def _get_condensed_question(self,
        project:str = "",
        user_id:str="",
        question:str="",
     ) -> str:
        """
        Return condensed question from conversation history and actual question
        -------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.
        system_msg - partial text which will be added at the begin of the system message
        user_id - user id
        question - actual question
            
        returns condensed question
        """
        chat_history = self.projects[project].memory.get_history(user_id)   # get a last conversation history   # MEM
        
        # when history is empty then question can't be condensed
        if len(chat_history) == 0:
            return question
 

        CONDENSE_Q_SYSTEM_PROMPT = """S ohledem na historii konverzace a nejnovější uživatelskou otázku, která by mohla odkazovat na historii konverzace, \
formulujte samostatnou otázku, které lze porozumět i bez historie konverzace. \
NEODPOVÍDEJTE na otázku, pouze ji v případě potřeby přeformulujte a jinak ji vraťte tak, jak je."""

        '''
        CONDENSE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        '''
        condense_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONDENSE_Q_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        
        condense_q_chain = condense_q_prompt | self.llm_condensed | StrOutputParser()

        condensed_question = condense_q_chain.invoke(
            {
                "chat_history": chat_history,
                "question": question,
            }
        )

        return condensed_question


    # @print_durations() 
    def _get_routing_value(self,
        project:str = "",
        question:str="",
        id_text:str="UCEL"
     ) -> str:
        """
        Return routing value for metadata fiel
        -------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.
        system_msg - partial text which will be added at the begin of the system message
        user_id - user id
        question - actual question
            
        returns condensed question
        """
        routing_field = self.projects[project].routing_field     
        if routing_field == None:
            return None

        text_data = self.projects[project].get_routing_text(id_text)
        if text_data == None:
            return None

        TEMPLATE = """Vyhledej hodnotu položky "{routing_field}" z níže zadaného kontextu, která nejlépe odpovídá zadané otázce.
{format_instructions}

Kontext:

{context}

#####
Otázka: {question}

Přísně dodržuj výstupní JSON formát.
"""
        try:
            # Define your desired data structure.
            class FieldRouter(BaseModel):
                """ Hodnota nalezené položky z kontextu """                
                cccc: str = Field(description=f"Hodnota položky {routing_field}")


            # Set up a parser + inject instructions into the prompt template.
            parser = JsonOutputParser(pydantic_object=FieldRouter)

            prompt = PromptTemplate(
                template = TEMPLATE,
                input_variables=["question"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

            chain = prompt | self.llm_condensed | parser

            answer = chain.invoke({"routing_field":routing_field, "context":text_data, "question": question})

            return answer.get("cccc")
        except Exception as e:
            return None     # when JSON isn't well formated then returns None
        
  


    def _set_retriever(self,
        project:str,
        ):
        """
        Set project's langchain retrievers
        -------------------------------------------------------------------------
        project - project name (is collection name in vector db). If is empty then set all retrievers
        """
        
        if project != "":
            project_list = [project]
        else:
            project_list = self.projects.keys()


        for project in project_list:
            # setting llm
            if self.projects[project].llm == None:
                self.projects[project].set_llm()

            # setting vectorstore
            if self.projects[project].vectorstore == None:
                self.projects[project].set_vectorstore(
                    db_type = self.db_type,
                    db_client = self.db_client,
                    embeddings = self.embeddings,
                    )

            # setting retrievers
            self.projects[project].set_retriever(
                verbose = self.verbose,             
                )


    def _get_retriever(self,
        project:str,
        ):
        """
        Return langchain retriever.
        -------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.

        returns retriever
        """

        retriever = self.projects[project].retriever_ensemble
        if retriever != None:
            return retriever

        for retriever, weight in zip(self.projects[project].retriever_set, self.projects[project].retriever_weights):
            if retriever != None and weight > 0:
                return retriever

        return None


   
    
    def _get_citation_chain(self,
        response:dict,
        citation_field:str = "source",
        ) -> str:
        """
        Return reference list to web pages. Maximum citations = 1.
        For example
        
        Další informace:
        1. https://www.multima.cz
        2. https://www.multima.cz/mentor
        3. https://www.keymate.cz
        -------------------------------------------------------------------------
        response - response object from ConversationalRetrievalChain()
        citation_field - citation field in metadata, which is used for citation

        returns text citation list
        """
        max_citations = 1        

        if response == None:
            return ""

        reference_list = [] 
        
        # select web references with order
        for item in response["documents"]:
            ref = item.get(citation_field)
            if ref and (ref not in reference_list):
                reference_list.append(ref)

        # create reference text
        citation_text = ""
        for row, item in enumerate(reference_list, 1):
            citation_text += f"\n{row}. {item}"

            if row >= max_citations:
                break;

        if citation_text != "":
            citation_text = "\nDalší informace:" + citation_text

        return citation_text


    # @print_durations() 
    def _get_project_par(self,
        project:str = "",
    )->tuple:
        """
        Return base parameters of the project from self.projects[].
        If self.projects[] not exists then setup it from environment variables
        -------------------------------------------------------------------------
        project - project name. Is mandatory.

        returns project item
        """
        if project not in self.projects:
            item = Project(
                project = project,
                collection = project,
                system_msg = self.system_msg,
                api_model = os.getenv("OPENAI_API_MODEL_GPT"),
                add_answer_time = self.answer_time,
                add_citation = False,
                self_doc_descr = "",
                self_metadata = [],
                metadata_parent_field = "",
                k_history = self.k_history,
                time_limit_history = self.time_limit_history,
                )            

            item.set_memory()   # set conversation memory

            item.set_llm()      # set LLM
            
            item.set_vectorstore(
                db_type = self.db_type,
                db_client = self.db_client,
                embeddings = self.embeddings,
            )
            
            item.set_retriever(
                verbose = self.verbose,     
            )

            self.projects[project] = item

        return self.projects[project]
    
     
    def _filter_error(self,
        error,
    ) -> str:
        """
        Create response from GPT filter
        -------------------------------------------------------------------------
        error - BadRequesrError (JSON with filter information)
        """
        str_error = str(error)

        # remove string part before {
        begin_position = str_error.find('{')
        if begin_position > 0:
            str_error = str_error[begin_position:]
            
        dict_error = ast.literal_eval(str_error)

        content_filter_result = dict_error["error"]["innererror"]["content_filter_result"]
        filtered_items = [key for key, value in content_filter_result.items() if value["filtered"]]
        
        bad_content = ""
        for item in filtered_items:
            match item:
                case "hate":
                    bad_content += "nenávist, "
                case "self-harm":
                    bad_content += "sebepoškozování, "
                case "sexual":
                    bad_content += "sex, "
                case "violence":
                    bad_content += "násilí, "

        bad_content = bad_content.rstrip()[:-1]

        return f"Vámi zadaný dotaz/výzva má závadný obsah: {bad_content}. Změňte prosím formulaci."

    def get_list_from_metadata(self,
        project:str = "",
        metadata_field = None,               
        ) -> list[str]:
        '''
        Get list values from field in self.documents.metadata
        Args:
            project - project name (collection name in Qdrant database)
            metadata_field: field name in metadata
        '''

        result = self.db_client.scroll(
            collection_name=project,
            limit=50000,
            with_payload=True,
            with_vectors=False,
        )
            
        records = result[0]
            
        metadatas = [item.payload["metadata"] for item in records]

        value_set = set()

        for metadata in metadatas:
            if metadata_field not in metadata:
                continue
            
            value = metadata[metadata_field].strip()
            if value == "":
                continue

            value_set.add(value)

        return [value for value in value_set]
    
    def get_projects(self,
        ) -> list[str]:
        '''
        Get list projects from self.projects
        Args:
        '''
        return list(self.projects.keys())


    def get_users(self,
        ) -> list:
        '''
        Get list users info from history
        Args:
        '''
        data = []
        
        for project in list(self.projects.values()):
            data.append({project.project:project.memory.get_users()})
            
        return data
         