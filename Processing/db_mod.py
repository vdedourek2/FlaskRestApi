''' 
KBADatabase - Class for comunication with PostgeSQL database needed for KBA

Sources:
https://www.linkedin.com/pulse/build-qa-bot-over-private-data-openai-langchain-leo-wang/
https://betterprogramming.pub/build-a-chatbot-for-your-notion-documents-using-langchain-and-openai-e0ad7b903b84

Library instalation:
pip install load-dotenv       # environment variable service
pip install psycopg2          # PostgreSQL database API
pip install SQLAlchemy        # The Python SQL Toolkit and Object Relational Mapper
'''

import os
from dotenv import load_dotenv      # python-dotenv
import psycopg2
import sqlalchemy.pool as pool

def getconn():
    return psycopg2.connect(
        user=os.getenv("SQLDB_UID"),
        password=os.getenv("SQLDB_PWD"),
        host=os.getenv("SQLDB_HOST"),
        dbname=os.getenv("SQLDB_DATABASE"))



class KBADatabase(object):
    '''
    ### Class for comunication with PostgeSQL database needed for KBA
    -----------------------------------------------------------------------
    '''
    
    projects:dict = {}   # projects parameters dictionary { "project_name": id_project }
    conn_pool: pool.QueuePool

    def __init__(self,
        ):
 
        load_dotenv()
       
        # SQL database connection
        try:
            self.conn_pool = pool.QueuePool(
                creator = getconn,
                max_overflow=10,
                pool_size=5,
                )
            
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while connecting to PostgreSQL", error)
            
    def __del__(self) -> None:
        self.conn_pool.dispose()
            
    def write_db_log(self,
        project:str = "",
        id_project:int = None,
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
        
        sql = """
            INSERT INTO public."PROJECTS_USER_HISTORY" ("ID_PROJECT", "QUESTION", "CONDENSED_QUESTION", "ANSWER", "ID_USER", "ELAPSED_TIME",
                                    "PROMPT_TOKENS", "COMPLETION_TOKENS", "TOTAL_COST")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            
        data = (id_project, question, condensed_question, answer, user_id, elapsed_time, prompt_tokens, completion_tokens, total_cost)
        self._write_db(sql, data)



    def write_db_protocol(self,
        project:str = "",
        protocol: str ="",      
  
    ):
        """
        Write protocol record to SQL db
        -------------------------------------------------------------------------
        project - project name. Is mandatory. If is empty then it don't rely on project
        protocol - record of protocol
        """
        if project == "":
            id_project = None
        else:
            id_project = self.get_project_id(project)

        sql = """
            INSERT INTO public."PROJECTS_PROTOCOL" ("ID_PROJECT", "PROTOCOL")
            VALUES (%s, %s);
            """        
         
        data = (id_project, protocol)
        self._write_db(sql, data)            
            
 
        
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
        sql = """
            DELETE FROM public."PROJECTS_RETRIEVER"
            WHERE "ID_PROJECT" = %s and "RETRIEVER_NAME" = %s;
            """
        data = (id_project, retriever_name,)
        self._write_db(sql, data)

        # insert new data
        sql = """
            INSERT INTO public."PROJECTS_RETRIEVER" ("ID_PROJECT", "RETRIEVER_NAME", "RETRIEVER_DATA")
            VALUES (%s, %s, %s);
            """
        data = (id_project, retriever_name, retriever_data)
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
 
        sql = 'select "RETRIEVER_DATA" from public."PROJECTS_RETRIEVER" where "ID_PROJECT" = %s and "RETRIEVER_NAME" = %s;'
        data = (id_project, retriever_name, )
        retriever_data = self._read_value(sql, data)

        return retriever_data



    def get_project_id(self,
        project:str = "",
    ):
        """
        Get project ID. If isn't setup in self.projects then is setup.
        -------------------------------------------------------------------------
        project - project name. Is mandatory.
        return project ID
        """
        
        if project in self.projects:
            return self.projects[project]
        
        sql = 'select "ID_PROJECT" from public."PROJECTS" where "PROJECT" = %s;'
        data = (project,)
        
        id_project = self._read_value(sql, data)
        if id_project != None:
            self.projects[project] = id_project
        
        return id_project
    

    def _write_db(self,
        sql:str, 
        data:tuple,
    ):
        """
        Write data to DB table
        -------------------------------------------------------------------------
        sql - SQL command
        data - tuple with columns
        """
        if not self.conn_pool:
            return
   
        # connection to DB
        try:
            conn = self.conn_pool.connect()
        except Exception as e:
            print(f"Database SQL exception: {e}")
            return

        cur = conn.cursor()
    
        # executing SQL
        cur.execute(sql, data)

        cur.close()
        conn.commit()
        conn.close()

    def _read_value(self,
        sql:str, 
        data:tuple,
    ):
        """
        Read value from DB table
        -------------------------------------------------------------------------
        sql - SQL command
        data - tuple with columns
        """
        value = None        

        if not self.conn_pool:
            return value
   
        # connection to DB
        try:
            conn = self.conn_pool.connect()
        except Exception as e:
            print(f"Database SQL exception: {e}")
            return value

        cur = conn.cursor()
        cur.execute(sql, data)
        rows = cur.fetchone()
        cur.close()
        conn.close()
            
        if type(rows) == tuple:
            if len(rows) > 0:
                value = rows[0]
  
        return value
    



 