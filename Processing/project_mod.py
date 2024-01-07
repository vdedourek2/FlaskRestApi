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
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.vectorstores import Chroma, Qdrant

from Processing.db_mod import KBADatabase
from Processing.parent_retriever_mod import QdrantParentRetriever


@dataclass
class Project:
    """Class for properties of chatbot project for Knowledge Base Assistant."""
    system_msg: str         # system message pro cht GPT
    api_model: str          # api model for query/qnswer model (for azure it is deployment)
    add_answer_time: bool   # True - answer is with elapsed time,  False - answer is without elapsed time
    add_citation:bool       # True - at the end of answer add web page references, False - without web page references
    self_doc_descr:str      # document description for Self Retriever
    self_metadata:list      # metadata description list for Self Retriever
    metadata_parent_field:str # Metadata field for parent doc
    k: int = 5              # number of chunks retrieved from a vector database

    llm:BaseChatModel = None          # LLM for query/answer
    vectorstore:any = None  # vectorstore Qdrant, Chromadb
    
    # weight vector of ensemble retriever. Weight are in interval <0, 1>
    # vector of (embedding retriever, SelfQueryRetriever, BM25, MultiQueryRetriever, ParentRetriever + SelfQueryRetriever)
    retriever_weights:tuple = (1, 0, 0, 0, 0)             # vector of initialized weights
    retriever_set:tuple = (None, None, None, None, None)  # vector of initialized retrievers 
    retriever_ensemble:any = None   # ensemble retriever (only if exists {weight : 0 < weight < 1} )


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

    def set_vectorstore(self,
        project:str,
        db_type:str,
        db_client,
        embeddings,
        ):
        """
        Set vectorstore
        --------------------------
        project - project name (is collection name in vector db). Is mandatory.
        db_type - Select option: 
            local - local Chroma DB in db directory, 
            qdrant - Qdrant database. Needs environment variables: QDRANT_URL, QDRANT_API_KEY
        db_client - Chroma client, Qdrant client
        embeddings - embeddings

        """
        match db_type:        
            case "local":
                self.vectorstore = Chroma(client = db_client, collection_name = project, embedding_function=embeddings)
            case "qdrant":
                self.vectorstore = Qdrant(client = db_client, collection_name = project, vector_name = "vector_embed", embeddings=embeddings)      


    def set_retriever(self,
        project:str = None,
        verbose:bool = False,
        ):
        """
        Set project's langchain retrievers:
        retriever_set
        retriever_ensemble
        -------------------------------------------------------------------------
        project - project name (is collection name in vector db). Is mandatory.
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
            
            output_parser = StructuredQueryOutputParser.from_components()
            query_constructor = prompt | self.llm | output_parser          

            self.retriever_set[1] = SelfQueryRetriever(
                search_type="mmr",
                query_constructor=query_constructor,
                vectorstore = self.vectorstore,
                structured_query_translator=QdrantTranslator(metadata_key = self.vectorstore.metadata_payload_key),
                verbose = verbose,
                # enable_limit=True,
            )
                
            if self.retriever_weights[1] == 1:
                return            

          
        # 2 - BM25 retriever
        # ************************
        if self.retriever_weights[2] > 0: 
            # read BM25 from database
            db = KBADatabase()
            bm25_data = db.read_retriever(project, "BM25")  # read retrieve object fron SQL database          
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
                k = self.k)
                
            if self.retriever_weights[4] == 1:
                return    



 
        # Setup ensemble retriever
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
        self_condition = None

        # 1 - SelfQueryRetriever            
        # ************************
        if self.retriever_set[1] != None:
            prompt = get_query_constructor_prompt(
                self.self_doc_descr,
                self.self_metadata,
            )
            
            output_parser = StructuredQueryOutputParser.from_components()
            query_constructor = prompt | self.llm | output_parser
            
            self_condition = query_constructor.invoke(
                {
                    "query": query
                }
            )
            
        return self_condition
        
   



