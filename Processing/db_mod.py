''' 
KBADatabase - Class for comunication with PostgeSQL database needed for KBA

Sources:
https://www.linkedin.com/pulse/build-qa-bot-over-private-data-openai-langchain-leo-wang/
https://betterprogramming.pub/build-a-chatbot-for-your-notion-documents-using-langchain-and-openai-e0ad7b903b84

Library instalation:
pip install load-dotenv       # environment variable service
pip install psycopg2          # PostgreSQL database API
pip install SQLAlchemy        # The Python SQL Toolkit and Object Relational Mapper
pip install funcy             # funcy library
'''

import os
from dotenv import load_dotenv      # python-dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from funcy import print_durations


class KBADatabase(object):
    '''
    ### Class for comunication with PostgeSQL database needed for KBA
    -----------------------------------------------------------------------
    '''
    
    projects:dict = {}   # projects parameters dictionary { "project_name": id_project }

    def __init__(self,
        ):
 
        load_dotenv()
        
        username = os.getenv("SQLDB_UID")
        password = os.getenv("SQLDB_PWD")
        host = os.getenv("SQLDB_HOST")
        port = 5432
        database = os.getenv("SQLDB_DATABASE")

        # Construct the PostgreSQL connection URL
        database_url = f'postgresql://{username}:{password}@{host}:{port}/{database}'

        # Set pool size based on your requirements
        pool_size = 5

        # Create engine with connection pool
        self.engine = create_engine(database_url, poolclass=QueuePool, pool_size=pool_size)
    
  
    # @print_durations()    
    def write_db_log(self,
        project:str = "",
        user_id: str ="",        # user id
        question:str = "",
        condensed_question:str = None,
        answer:str="",
        api_model:str="",
        elapsed_time:float = 0,
        prompt_tokens:int = None,
        completion_tokens:int = None,
        total_cost:float = None,
    ):
        """
        Write log information question/answer to SQL db
        -------------------------------------------------------------------------
        project - project name. Is mandatory.
        id_project - project ID. If is None then is find out in database
        user_id - unique user id (is mandatory)
        question - question
        condensed_question - condensed question (reformulating input question from the chat history)
        answer - answer
        api_model - model of the ChatGPT API. (if empty then environment variable "OPENAI_API_MODEL_GPT" is used)
            For open_ai: gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613
                         gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613
            For azure: deployment name         
        elapsed_time - elapsed time in seconds
        prompt_tokens - Number of prompt tokens used in ChatCompletion operation
        completion_tokens - Number of completion tokens used in ChatCompletion operation
        total_cost - Total cost of the ChatCompletion operation in USD
        """ 
        id_project = self.get_project_id(project)
  
        if id_project == None:
            return
        
        sql = """INSERT INTO public."PROJECTS_USER_HISTORY" ("ID_PROJECT", "QUESTION", "CONDENSED_QUESTION", "ANSWER", "ID_USER", "ELAPSED_TIME",
"PROMPT_TOKENS", "COMPLETION_TOKENS", "TOTAL_COST")
VALUES (:id_project, :question, :condensed_question, :answer, :user_id, :elapsed_time, :prompt_tokens, :completion_tokens, :total_cost);"""
            
        data = {"id_project":id_project, "question":question, "condensed_question":condensed_question, "answer":answer,
                "user_id":user_id, "elapsed_time":elapsed_time, "prompt_tokens":prompt_tokens, 
                "completion_tokens":completion_tokens, "total_cost":total_cost}

        self._write_db(sql, data)



    def write_db_protocol(self,
        project:str = "",
        protocol: str ="",      
  
    ):
        """
        Write protocol record to SQL db PROJECTS_PROTOCOL
        -------------------------------------------------------------------------
        project - project name. Is mandatory. If is empty then it don't rely on project
        protocol - record of protocol
        """
        if project == "":
            id_project = None
        else:
            id_project = self.get_project_id(project)

        sql = """INSERT INTO public."PROJECTS_PROTOCOL" ("ID_PROJECT", "PROTOCOL")
VALUES (:id_project, :protocol);"""   
        data = {"id_project":id_project, "protocol":protocol}

        self._write_db(sql, data)            
            
    def get_db_texts(self,
        project: str = "",
        id_text: str ="",      
    )->str:
        """
        Read text record to SQL db PROJECTS_TEXTS
        -------------------------------------------------------------------------
        project - project name. Is mandatory. 
        id_text - text id
        
        return:
        text_data - text data
        """
        id_project = self.get_project_id(project)
 
        sql = 'select "TEXT_DATA" from public."PROJECTS_TEXTS" where "ID_PROJECT" = :id_project and "ID_TEXT" = :id_text;'
        data = {"id_project":id_project, "id_text":id_text}

        text_data = self.read_value(sql, data)

        return text_data          
            
    def write_db_texts(self,
        project: str = "",
        id_text: str ="",      
        text_data: str ="",      
        description: str ="",      
    ):
        """
        Write text record to SQL db PROJECTS_TEXTS
        -------------------------------------------------------------------------
        project - project name. Is mandatory. 
        id_text - text id
        text_data - text data
        description - description of the text
        """
        id_project = self.get_project_id(project)
 
        sql1 = 'DELETE from public."PROJECTS_TEXTS" where "ID_PROJECT" = :id_project and "ID_TEXT" = :id_text;'
        data = {"id_project":id_project, "id_text":id_text}

        self._write_db(sql1, data)                 
 
        sql2 = """INSERT INTO public."PROJECTS_TEXTS" ("ID_PROJECT", "ID_TEXT", "TEXT_DATA", "DESCRIPTION")
VALUES (:id_project, :id_text, :text_data, :description);"""        
        data = {"id_project":id_project, "id_text":id_text, "text_data":text_data, "description":description}

        self._write_db(sql2, data)            
            



        
    def write_retriever(self,
        project:str,
        retriever_name:str,
        retriever_data,      
    ):
        """
        Write retriever object data to PROJECTS_RETRIEVER
        -------------------------------------------------------------------------
        project - project name. Is mandatory. 
        retriever_name - retriever name. BM25 - BM25 retriever, PARENT - parent document retriever
        retriever_data - retriever data
        """
        id_project = self.get_project_id(project)
 
        # deleting original data
        sql = """DELETE FROM public."PROJECTS_RETRIEVER"
WHERE "ID_PROJECT" = :id_project and "RETRIEVER_NAME" = :retriever_name;"""
        data = {"id_project":id_project, "retriever_name":retriever_name}
        
        self._write_db(sql, data)

        # insert new data
        sql = """INSERT INTO public."PROJECTS_RETRIEVER" ("ID_PROJECT", "RETRIEVER_NAME", "RETRIEVER_DATA")
VALUES (:id_project, :retriever_name, :retriever_data);"""
        data = {"id_project":id_project, "retriever_name":retriever_name, "retriever_data":retriever_data}
        
        self._write_db(sql, data)
        
    def read_retriever(self,
        project:str,
        retriever_name:str,
    ):
        """
        Read retriever object data from PROJECTS_RETRIEVER
        -------------------------------------------------------------------------
        project - project name. Is mandatory. 
        retriever_name - retriever name. BM25 - BM25 retriever, PARENT - parent document retriever
        """
        id_project = self.get_project_id(project)
        
        sql = 'select "RETRIEVER_DATA" from public."PROJECTS_RETRIEVER" where "ID_PROJECT" = :id_project and "RETRIEVER_NAME" = :retriever_name;'
        data = {"id_project":id_project, "retriever_name":retriever_name}

        retriever_data = self.read_value(sql, data)

        return retriever_data



    def get_project_id(self,
        project:str = "",
    ):
        """
        Get project ID. If isn't setup in self.projects then is setup.
        If doesn't exist in PROJECTS, the in inserted there
        -------------------------------------------------------------------------
        project - project name. Is mandatory.
        return project ID
        """
        # if project is presented in self.projects then return id_project
        if project in self.projects:
            return self.projects[project]
        
        sql = 'select "ID_PROJECT" from public."PROJECTS" where "PROJECT" = :project;'
        data = {"project":project}
        
        id_project = self.read_value(sql, data)
        
        if id_project != None:
            self.projects[project] = id_project
            return id_project
        
        sql = 'INSERT INTO public."PROJECTS" ("PROJECT") VALUES (:project) RETURNING "ID_PROJECT";'        
        data = {"project":project}

        id_project = self.read_value(sql, data)       

        return int(id_project)
    

    def _write_db(self,
        sql:str, 
        data:dict,
    ):
        """
        Write data to DB table
        -------------------------------------------------------------------------
        sql - SQL command
        data - tuple with columns
        """
        try:
            # connection to DB
            conn = self.engine.connect()
            conn.execute(text(sql), data)
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Database SQL exception: {e}")
            return

    def read_value(self,
        sql:str, 
        data:dict,
    ):
        """
        Read value from DB table
        -------------------------------------------------------------------------
        sql - SQL command
        data - tuple with columns
        """
        try:
        # connection to DB
            conn = self.engine.connect()
            rows = conn.execute(text(sql), data).fetchone()
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Database SQL exception: {e}")
            return None
         
        if rows:
            value = rows[0]
        else:
            value = None
  
        return value
    
    def read_values(self,
        sql:str, 
        size:int,
        data:dict,
    ):
        """
        Read value from DB table
        -------------------------------------------------------------------------
        sql - SQL command
        data - tuple with columns
        """
    
        try:
            # connection to DB
            conn = self.engine.connect()
            rows = conn.execute(text(sql), data).fetchmany(size = size)
            conn.commit()
            conn.close()
  
        except Exception as e:
            print(f"Database SQL exception: {e}")
            return None
  
        return rows
    
    



 