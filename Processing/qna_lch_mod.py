# Requests:
# 1 - Initialize history for user_id + cearing -  OK
# 2 - Implement recursive web crawler - OK
# 3 - Implement getting metadata information for result answer
# 4 - multiproject solution - OK
# 5 - question/answer logging - OK
# 6 - connection pool - OK
#-------------------------------------------------------------
# Challenges:
# - attack protection
# - price meassuring


''' 
KBAQnA - Class for talking with Knowledge Assistent on Vectore database data
        which are created by KBAIndex class (based on Langchain)

Sources:
https://www.linkedin.com/pulse/build-qa-bot-over-private-data-openai-langchain-leo-wang/
https://betterprogramming.pub/build-a-chatbot-for-your-notion-documents-using-langchain-and-openai-e0ad7b903b84

Library instalation:
pip install load-dotenv       # environment variable service
pip install langchain         # framework for LLM
pip install openai            # OpenAI
pip install chromadb          # Chromadb database API
pip install qdrant-client     # Qdrant database API
pip install psycopg2          # PostgreSQL database API
pip install SQLAlchemy        # The Python SQL Toolkit and Object Relational Mapper
pip install lark              # Needed for SelfQuerying
'''

from dotenv import load_dotenv      # python-dotenv
import os
import time
import ast
from openai import OpenAI, AzureOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.vectorstores import Chroma, Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from qdrant_client import QdrantClient
import chromadb
import psycopg2
import sqlalchemy.pool as pool

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def getconn():
    return psycopg2.connect(
        user=os.getenv("SQLDB_UID"),
        password=os.getenv("SQLDB_PWD"),
        host=os.getenv("SQLDB_HOST"),
        dbname=os.getenv("SQLDB_DATABASE"))


class KBAQnA(object):
    '''
    ### Class for talking with Knowledge Assistent on Vectore database data,
        which are created by KBAIndex class (based on Langchain)
    -----------------------------------------------------------------------
    db_type - Select option: 
        local - local Chroma DB in db directory, 
        qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
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
        k:            int  = 3,
        k_history:    int  = 3,
        time_limit_history: int = 3600,
        verbose:      bool = False,
        answer_time:  bool = False,
        ):
 
        load_dotenv()
        
        self.db_type            = db_type
        self.db_dir             = db_dir
        self.system_msg         = system_msg
        self.k                  = k
        self.k_history          = k_history
        self.verbose            = verbose
        self.answer_time        = answer_time
        self.time_limit_history = time_limit_history
        # self.self_doc_descr     = self_doc_descr
        # self.self_metadata      = self_metadata
         
        # vector database connection
        if db_type == "local":
            self.chroma_client = chromadb.PersistentClient(path=db_dir)

        if db_type == "qdrant":
            self.qdrant_client =    QdrantClient(
                url = os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                prefer_grpc=True,               
                )  
            
        # SQL database connection
        try:
            self.conn_pool = pool.QueuePool(
                creator = getconn,
                max_overflow=10,
                pool_size=5,
                )
            
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while connecting to PostgreSQL", error)
            
        self.history  = []   # chat history [("project":project, "user_id":user_id, "last_time":last_time, "history":[(question, answer),...]), ...]
        self.projects = {}   # projects parameters dictionary {"project_name": 
                             #   {"id_project":id_project, "system_msg":system_msg, "api_model":api_model, "answer_time":answer_time, "citation":citation},
                             #     "self_doc_descr":self_doc_descr, "self_metadata: : self_metadata
                             #   }
                             #
        
    def set_cls_par(self,
        db_type:str = "",
        db_dir:str = "",
        system_msg:str  = "",
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
        
        history = self.history
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
        
        if not erase_history:
             self.history = history

        self.projects = projects

    def get_cls_par(self):
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
        self_doc_descr: str = None,
        self_metadata:list = None,
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
            For azure: deployment name
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time (if is None then is unchanged)
        citation - True - at the end of answer add web page references, False - without web page references (if is None then is unchanged)
        self_doc_descr - document description for Self Retriever
        self_metadata - metadata description list for Self Retriever (if isn't empty then is used Self Retriever)
        erase_history - True - erase history self.history for the project, False - history isn't erased
        '''       
        
        self._get_project_par(project)
        
        item = self.projects[project]

        # check correct model (only for OpenAi)
        if api_model != "":
            if ( os.getenv("OPENAI_API_TYPE") == "open_ai"):
                client = OpenAI()
                if api_model in [item.id for item in client.models.list().data]:
                    item["api_model"] = api_model
                else:
                    print(f"Error in set_project_par: Model '{api_model}' doesn't exist")
            else:
                client = AzureOpenAI(
                    api_key = os.getenv("OPENAI_API_KEY"),
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),                    
                    )

        if system_msg != "":
            item["system_msg"] = system_msg
        
        if answer_time != None:
            item["answer_time"] = answer_time
  
        if citation != None:
            item["citation"] = citation

        if self_doc_descr != None:
            item["self_doc_descr"] = self_doc_descr
        
        if self_metadata != None:
            item["self_metadata"] = self_metadata
  
        self.projects[project] = item

        if erase_history:
            for item in self.history.copy():
                if item["project"] == project:
                    self.history.remove(item)             

    def get_project_par(self,
        project:str = "",
                        
    ):
        '''
        Get project parameters.
        ----------------------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.

        Parameters:
        system_msg - partial text which will be added at the begin of the system message 
        api_model - model of the ChatGPT API.
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        answer_time - True - answer is with elapsed time,  False - answer is without elapsed time
        citation - True - at the end of answer add web page references, False - without web page references
        '''       
        
        if project not in self.projects:
            return {}

        item = self.projects[project]
        
        return {
            "system_msg":       item["system_msg"],
            "api_model":        item["api_model"],
            "answer_time":      item["answer_time"],
            "citation":         item["citation"],
            "self_doc_descr":   item["self_doc_descr"],
            "self_metadata":    item["self_metadata"],
            }


    
    def answer_question(self,
        question:str ="Co je Keymate?",
        user_id: str ="",        # user id
        project:str = "",
        system_msg:str = "",
        api_type:str = "",
        api_base:str = "",
        api_key:str = "",
        api_version:str = "",
        api_model:str="",
     ):
        """
        Answer a question 
        -------------------------------------------------------------------------
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
 
        returns answer
        """
        st = time.time()
 
        history = self._get_history(project, user_id)   # get a last conversation history
        
        # getting project parameters
        (id_project, pr_system_message, pr_api_model, pr_answer_time, citation, self_doc_descr, self_metadata) = self._get_project_par(project)

        if system_msg == "":
            system_msg = pr_system_message

        if api_model == "":
            api_model = pr_api_model

        qa = self._get_qa(project, system_msg, api_type, api_base, api_key, api_version, api_model, self_doc_descr, self_metadata)

        try:
            response = qa({"question": question, "chat_history": history})  # get answer
            answer = response["answer"]
        except Exception as e:
            response = None
            answer = self._filter_error(e)

        self._add_history(project, user_id, question, answer)   # save question/answer to the conversation history

        et = time.time()
        
        # write question/answer to DB log
        self._write_db_log(
            project = project,
            question = question,
            answer=answer,
            api_model = api_model,
            elapsed_time = et - st,
        )

        if citation:
            answer += self._get_citation(response)


        if pr_answer_time:
            answer += f" ({round(et - st, 3)} s)"
            
        return answer

    def get_dataset_item(self,
        question:str ="Co je Keymate?",
        project:str = "",
        ):
        """
        Generate dataset item for one question. It is used in RAGAS.

        Before running should be setup  set_project_par with citatio=True.
        Generating doesn't work with history
        -------------------------------------------------------------------------
        question - question (is mandatory)
        project - project name (is collection name in vector db). Is mandatory.
 
        returns data_item = {'question': question, 'answer': answer, 'contexts': []}  
        """
 
        # getting project parameters
        (id_project, pr_system_message, pr_api_model, pr_answer_time, citation, self_doc_descr, self_metadata) = self._get_project_par(project)

        system_msg = pr_system_message
        api_type = ""
        api_base = ""
        api_key = ""
        api_version = ""
        api_model = pr_api_model

        qa = self._get_qa(project, system_msg, api_type, api_base, api_key, api_version, api_model, self_doc_descr, self_metadata)

        try:
            response = qa({"question": question, "chat_history": []})  # get answer
            answer = response["answer"]
            contexts = [doc.page_content for doc in response["source_documents"]]
        except Exception as e:
            answer = self._filter_error(e)
            contexts = []
        
        # source is in doc.metadata["source"]
            
        return {'question': question, 'answer': answer, 'contexts': contexts}    





    def _get_history(self,
        project: str ="",        # project
        user_id: str ="",        # user id
    ):
        """
        Get history for user
        chat history -> self.history
         [("project":project, "user_id":user_id, "last_time":last_time, "history":[(question, answer),...]), ...]
        """
        
        # find index user item
        index = None
        for i, element in enumerate(self.history): 
            if element["project"] == project and element["user_id"] == user_id:
                index = i
                break

        # add new user
        if index == None:
            history = []
        else:
            # update item
            history = self.history[index]["history"]

        return history


    def _add_history(self,
        project: str ="",        # project
        user_id: str ="",        # user id
        question:str="",
        answer:str="",
    ):
        """
        Add question, answer to history.
        Shrink the history to self.k_history length.
        Clearing the history after the time interval.
        
        chat history -> self.history
         [("project":project, "user_id":user_id, "last_time":last_time, "history":[(qusetion, answer),...]), ...]
        """
        # Clearing the history after the time interval.
        self.history[:] = [item for item in self.history if time.time() - item["last_time"] <= self.time_limit_history]

        # find index user item
        index = None
        for i, element in enumerate(self.history): 
            if element["project"] == project and element["user_id"] == user_id:
                index = i
                break
        
        if index == None:
            # add new user
            item = { "project":project, "user_id":user_id, "last_time":time.time(), "history":[(question, answer)]}
            self.history.append(item)
        else:
            # update item
            item = self.history[index]
            item["last_time"] = time.time()
            item["history"].append((question, answer))
            item["history"] = item["history"][-self.k_history:]
            self.history[index] = item
            

    def _get_qa(self,
        project:str = "",
        system_msg:str = "",
        api_type:str = "",
        api_base:str = "",
        api_key:str = "",
        api_version:str = "",
        api_model:str="",
        self_doc_descr:str="",
        self_metadata:list=[],
        
     ):
        """
        Return qa object for Question/answer
        -------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.
        system_msg - partial text which will be added at the begin of the system message
        api_type - OpenAI type - open_ai, azure (if empty then environment variable "OPENAI_API_TYPE" is used )
        api_base - URL base of the ChatGPT API (if empty then environment variable "OPENAI_API_BASE" is used 
        api_key - API key of the ChatGPT (if empty then environment variable "OPENAI_API_KEY" is used)
        api_version - version of the ChatGPT API (if empty then environment variable "OPENAI_API_VERSION" is used)
        api_model - model of the ChatGPT API. (if empty then environment variable "OPENAI_API_MODEL_GPT" is used)
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name 

        returns answer
        """
 
        if api_type == "":
            api_type = os.getenv("OPENAI_API_TYPE")
            
        if api_base == "":
            api_base = os.getenv("OPENAI_API_BASE") if api_type == "open_ai" else os.getenv("AZURE_OPENAI_ENDPOINT")
            
        if api_key == "":
            api_key = os.getenv("OPENAI_API_KEY") if api_type == "open_ai" else os.getenv("AZURE_OPENAI_API_KEY")
 
        if api_version == "":
            api_version = os.getenv("OPENAI_API_VERSION")

        # model definition for get question
        if (api_type == "open_ai"):
            embeddings = OpenAIEmbeddings(
                api_base = api_base,
                openai_api_key=api_key,
                api_version=api_version,
                )
            
            llm = ChatOpenAI(
                temperature=0,
                model_name=api_model,
                )
        else:
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("OPENAI_API_MODEL_ADA"),
                azure_endpoint=api_base,
                openai_api_version=api_version,
                )
            
            llm = AzureChatOpenAI(
                temperature=0,
                deployment_name=api_model,
                )

        if self.db_type == "local":
            vectorstore = Chroma(client = self.chroma_client, collection_name = project, embedding_function=embeddings)

        if self.db_type == "qdrant":
            vectorstore = Qdrant(client = self.qdrant_client, collection_name = project, embeddings=embeddings)

  
        # Initialise Langchain - Conversation Retrieval Chain 2
        # https://stackoverflow.com/questions/76240871/how-do-i-add-memory-to-retrievalqa-from-chain-type-or-how-do-i-add-a-custom-pr
        general_system_template = r""" 
Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}
        """

        if system_msg != "":
            general_system_template = system_msg + general_system_template

        general_user_template = "{question}"
        messages = [
                    SystemMessagePromptTemplate.from_template(general_system_template),
                    HumanMessagePromptTemplate.from_template(general_user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages( messages )
        
        if len(self_metadata) == 0:
            retriever = vectorstore.as_retriever(search_kwargs={'k': self.k})      # standard vector DB retriever
        else:
            retriever = SelfQueryRetriever.from_llm(    # vector DB retriever with filter on a DB metadata
                llm = llm,
                vectorstore = vectorstore,
                document_contents = self_doc_descr,
                metadata_field_info = self_metadata,
                verbose=self.verbose,
                )


        return ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = retriever,
            verbose=self.verbose,
            combine_docs_chain_kwargs={'prompt': qa_prompt},
            return_source_documents=self.projects[project]["citation"], 
        )
            

    def _get_citation(self,
        response    
        ):
        """
        Return reference list to web pages.
        For example
        
        Další informace:
        1. https://www.multima.cz
        2. https://www.multima.cz/mentor
        3. https://www.keymate.cz
        -------------------------------------------------------------------------
        response - responce object from ConversationalRetrievalChain()

        returns text citation list
        """
        if response == None:
            return ""

        reference_list = [] 
        
        # select web references with order
        for item in response["source_documents"]:
            ref = item.metadata["source"]
            if ref.startswith("https:") and ref not in reference_list:
                reference_list.append(ref)

        # create reference text
        citation_text = ""
        for row, item in enumerate(reference_list):
            citation_text += f"\n{row + 1}. {item}"

            if row >= 2:
                break;

        if citation_text != "":
            citation_text = "\nDalší informace:" + citation_text

        return citation_text

    def _get_project_par(self,
        project:str = "",
    ):
        """
        Return base parameters of the project (system_msg, api_model, answer_time) from self.projects[].
        If self.projects[] not exists then setup it from environment variables
        -------------------------------------------------------------------------
        project - project name. Is mandatory.

        returns (system_msg, api_model, answer_time, citation, self_doc_descr, self_metadata)
        """
 
        if project not in self.projects:
            self.projects[project] = {"id_project":None,
                                      "system_msg":self.system_msg,
                                      "api_model":os.getenv("OPENAI_API_MODEL_GPT"),
                                      "answer_time":self.answer_time,
                                      "citation":False,
                                      "self_doc_descr":"",
                                      "self_metadata":[],
                                      }
            
        item = self.projects[project]
        
        return (item["id_project"], item["system_msg"], item["api_model"], item["answer_time"], item["citation"], item["self_doc_descr"], item["self_metadata"])
    

    def _write_db_log(self,
        project:str = "",
        question:str = "",
        answer:str="",
        api_model:str="",
        elapsed_time:float = 0,
    ):
        """
        Write log information question/answer to SQL db
        -------------------------------------------------------------------------
        project - project name. Is mandatory.
        question - question
        answer - answer
        api_model - model of the ChatGPT API. (if empty then environment variable "OPENAI_API_MODEL_GPT" is used)
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        elapsed_time - elapsed time in seconds
        """ 
        if not self.conn_pool:
            return

        # getting project id
        id_project = self.projects[project]["id_project"]
            
        # if projects id isn't retrieved then read from DB
        if id_project == None:
            try:
                conn = self.conn_pool.connect()
            except Exception as e:
                print(f"Database SQL exception: {e}")
                return


            cur = conn.cursor()
            cur.execute('select "ID_PROJECT" from public."PROJECTS" where "PROJECT" = %s;', (project,))
            rows = cur.fetchone()
            cur.close()
            conn.close()
            
            if type(rows) == tuple:
                if len(rows) > 0:
                    id_project = rows[0]
                    self.projects[project]["id_project"] = id_project
  
        if id_project == None:
            return

        # insert log to PROJECTS_LOG
        try:
            conn = self.conn_pool.connect()
        except Exception as e:
            print(f"Database SQL exception: {e}")
            return

        cur = conn.cursor()
    
        sql = """
            INSERT INTO public."PROJECTS_LOG" ("ID_PROJECT", "QUESTION", "ANSWER", "CHATGPT_MODEL", "ELAPSED_TIME")
            VALUES (%s, %s, %s, %s, %s);
            """
        cur.execute(sql,
            (id_project, question, answer, api_model, elapsed_time,)
            )
        cur.close()
        conn.commit()
        conn.close()

    def _filter_error(self,
        error,
    ):
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