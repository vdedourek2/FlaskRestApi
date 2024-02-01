''' 
Project - Class for holding information about chatbot project
        which used by KBAQnA class (based on Langchain)

Library instalation:
pip install chromadb          # Chromadb database API
pip install qdrant-client     # Qdrant database API
pip install lark              # Needed for SelfQuerying
pip install langchain         # langchain framework
'''
from dataclasses import dataclass
import os
import pickle

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chains.query_constructor.base import StructuredQueryOutputParser, get_query_constructor_prompt
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.qdrant import QdrantTranslator
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from langchain_community.vectorstores import Chroma, Qdrant
from langchain_community.vectorstores.pgvector import PGVector

from Processing.db_mod import KBADatabase
from Processing.parent_retriever_mod import QdrantParentRetriever
from Processing.qna_memory_mod import KBAMemory
from Processing.pgvector_translator_mod import PgvectorTranslator

@dataclass
class Project:
    """Class for properties of chatbot project for Knowledge Base Assistant."""
    project:str                 # project name
    collection:str              # collection in vector database (for BM25 project in SQL database)
    system_msg: str             # system message pro cht GPT
    api_model: str              # api model for query/qnswer model (for azure it is a deployment)
    add_answer_time: bool       # True - answer is with elapsed time,  False - answer is without elapsed time
    self_doc_descr:str          # document description for Self Retriever
    self_metadata:list          # metadata description list for Self Retriever
    metadata_parent_field:str   # Metadata field for parent doc
    add_citation:bool = False   # True - at the end of answer add web page references, False - without web page references
    citation_field:str = "source"   # field in metadata, which is used for citation
    clone:bool = False          # True - clone project which was created from project <collection>, False - normal project (project = collection)
    k: int = 5                  # number of chunks retrieved from a vector database
    k_history:int = 3           # number of an messages in a history
    time_limit_history:int=1200 # garbage collection timeout of the user inactivity
    routing_field:str = None    # field in metadata, which is used as condition  (field = value) for getting context for RAG
    routing_text:str = None     # text which is used for routing to RAG model
    
    llm:BaseChatModel = None    # LLM for query/answer
    db_type:str = None          # db type for vectorstore
    vectorstore:any = None      # vectorstore Qdrant, Chromadb
    
    # weight vector of ensemble retriever. Weight are in interval <0, 1>
    # vector of (embedding retriever, SelfQueryRetriever, BM25, MultiQueryRetriever, ParentSelfQueryRetriever)
    retriever_weights:tuple = (1, 0, 0, 0, 0)             # vector of initialized weights
    retriever_set:tuple = (None, None, None, None, None)  # vector of initialized retrievers 
    retriever_ensemble:any = None   # ensemble retriever (only if exists {weight : 0 < weight < 1} )

    memory:KBAMemory = None     # Class for saving conversation with user

    def set_llm(self):
        """
        Set llm.
        -------------------------------------------------------------------------
        """ 
        # model definition for geting question
        if (os.getenv("OPENAI_API_TYPE") == "open_ai"):
            self.llm = ChatOpenAI(
                temperature=0,
                model = self.api_model,
                )
        else:
            self.llm = AzureChatOpenAI(
                temperature=0,
                deployment_name = self.api_model,
                )

    def set_memory(self):
        """
        Set conversation memory.
        -------------------------------------------------------------------------
        """ 
        self.memory = KBAMemory()
        self.memory.k_history = self.k_history                       # number of an messages in a history
        self.memory.time_limit_history = self.time_limit_history     # garbage collector timeout of the user inactivity in seconds


    def set_vectorstore(self,
        db_type:str,
        db_client,
        embeddings,
        ):
        """
        Set vectorstore
        --------------------------
        db_type - Select option: 
            local - local Chroma DB in db directory, 
            qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
            pgvector - pgvector extension in PostgreSQL. Needs variables:SQLDB_HOST, SQLDB_DATABASE, SQLDB_UID, SQLDB_PWD                    
        db_client - Chroma client, Qdrant client
        embeddings - embeddings

        """
        self.db_type = db_type
        
        try:
             
            # creating vectorstore
            match db_type:        
                case "local":
                    # check existence collection - if doesn't exist then is exception
                    collection_info = db_client.get_collection(name=self.collection, embedding_function=embeddings)  

                    self.vectorstore = Chroma(client = db_client, collection_name = self.collection, embedding_function=embeddings)

                case "qdrant":
                    # check existence collection - if doesn't exist then is exception
                    collection_info = db_client.get_collection(collection_name=self.collection)  
                    
                    self.vectorstore = Qdrant(client = db_client, collection_name = self.collection, vector_name = "vector_embed", embeddings=embeddings)      

                case "pgvector":
                    # https://github.com/langchain-ai/langchain/issues/9726
                    # https://github.com/langchain-ai/langchain/issues/13281
                    # check existence collection - if doesn't exist then is exception
                    collection_info = db_client.get_collection(collection_name=self.collection)  
                    
                    host= os.getenv('SQLDB_HOST')
                    port= "5432"
                    user= os.getenv("SQLDB_UID")
                    password= os.getenv("SQLDB_PWD")
                    dbname= os.getenv("SQLDB_DATABASE")
                    CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?client_encoding=utf8&sslmode=require"


                    # search_kwargs={'filter': { 'locale': 'en-US', "type": {"in": ["TYPE1", "TYPE2"]} # this filter works
                    self.vectorstore = PGVector(
                        collection_name=self.collection,
                        connection_string=CONNECTION_STRING,
                        embedding_function=embeddings,
                    )                    

        except Exception as e:
            raise Exception(e)


    def set_retriever(self,
        verbose:bool = False,
        ):
        """
        Set project's langchain retrievers:
        retriever_set
        retriever_ensemble
        -------------------------------------------------------------------------
        verbose - True - logging process question/answer, False - without logging
        """
 
        # setup retrievers
        self.retriever_set = [None, None, None, None, None]   # list of retrievers
        self.retriever_ensemble = None
        
        # 0 - embeddings retriever
        # ************************
        if self.retriever_weights[0] > 0:   
            self.retriever_set[0] = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': self.k},
                )      # standard vector DB retriever
                
            if self.retriever_weights[0] == 1:
                return            
  
        # 1 - SelfQueryRetriever            
        # ************************
        if self.retriever_weights[1] > 0 or self.retriever_weights[4] > 0:
             
            prompt = get_query_constructor_prompt(
                self.self_doc_descr,
                self.self_metadata,
            )
            
            output_parser = StructuredQueryOutputParser.from_components(fix_invalid = True)
            query_constructor = prompt | self.llm | output_parser

            match self.db_type:
                case "local"|"pgvector":
                    structured_query_translator = ChromaTranslator()
                # case "pgvector":
                #     structured_query_translator = PgvectorTranslator()
                case "qdrant":                    
                    structured_query_translator = QdrantTranslator(metadata_key = self.vectorstore.metadata_payload_key)
                case _:                  
                    structured_query_translator = None

            self.retriever_set[1] = SelfQueryRetriever(
                search_type="mmr",
                search_kwargs={'k': self.k},
                query_constructor=query_constructor,
                vectorstore = self.vectorstore,
                structured_query_translator=structured_query_translator,
                verbose = verbose,
                enable_limit=True,
            )
                 
            if self.retriever_weights[1] == 1:
                return            

          
        # 2 - BM25 retriever
        # ************************
        if self.retriever_weights[2] > 0: 
            # read BM25 from database
            db = KBADatabase()
            bm25_data = db.read_retriever(self.collection, "BM25")  # read retrieve object fron SQL database          
            self.retriever_set[2] = pickle.loads(bm25_data)
 
            self.retriever_set[2].set_database()
            self.retriever_set[2].k = self.k
 
            if self.retriever_weights[2] == 1:
                return            

            
        # 3 - MultiQueryRetriever
        # ************************
        if self.retriever_weights[3] > 0: 
            self.retriever_set[3] = MultiQueryRetriever.from_llm(
                retriever = self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={'k': self.k},
                    ),      # standard vector DB retriever
                llm = self.llm
                )
           
            self.retriever_set[3].verbose = verbose

            if self.retriever_weights[3] == 1:
                return 
            
        # 4 - SelfQueryRetriever + Parent retriever           
        # *****************************************
        if self.retriever_weights[4] > 0:
            self.retriever_set[4] = QdrantParentRetriever(
                vectorstore = self.vectorstore, 
                retriever = self.retriever_set[1],
                metadata_parent_field = self.metadata_parent_field,
                k = self.k,
            )
                
            if self.retriever_weights[4] == 1:
                return    



 
        # Setup ensemble retriever
        # *****************************************
        en_weights = []
        en_retrievers = []
        for index, weight in enumerate(self.retriever_weights):
            if weight > 0:
                en_weights.append(weight)
                en_retrievers.append(self.retriever_set[index])

        self.retriever_ensemble = EnsembleRetriever(
            retrievers = en_retrievers,
            weights=en_weights,
        )
        

    def get_self_condition(self,
        query:str,
        ):
        """
        get condition of SelfRetrieveru
 
        -------------------------------------------------------------------------
        Args:
        query - query
        Return:
            SelfRetriever condition
        """
        prompt = get_query_constructor_prompt(
            self.self_doc_descr,
            self.self_metadata,
        )
            
        output_parser = StructuredQueryOutputParser.from_components(fix_invalid = True)
        query_constructor = prompt | self.llm | output_parser
            
        self_condition = query_constructor.invoke( { "query": query } )
            
        return self_condition
        
    def get_routing_text(self,
        id_text:str,
        )->str:
        """
        get routing text from PROJECT_TEXTS
 
        -------------------------------------------------------------------------
        Args:
        id_text - ID_TEST in PROJECT_TEXTS
        
        Return:
            routing text
        """
        if self.routing_field == None:
            return None

        if self.routing_text == None:
            db = KBADatabase()
            self.routing_text = db.get_db_texts(project = self.project, id_text = id_text)
            
        return self.routing_text
        
   
   



