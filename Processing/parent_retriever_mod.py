''' 
QdrantParentRetriever - parent retriever based on Qdrant database and Langchain

Library instalation:
pip install qdrant-client     # Qdrant database API
pip install langchain         # langchain framework
'''
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class QdrantParentRetriever(BaseRetriever):
    """`Parent retriever with Qdrant."""

    vectorstore:Any
    """ Qdrant vectorstore"""  
    
    retriever:Any
    """ child retriever"""      

    metadata_parent_field:str =""
    """ Metadata field for parent doc."""
    
    k: int = 4
    """ Number of documents to return."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
   
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        docs = self.retriever.get_relevant_documents(query)
       
        documents = []

        for doc in docs:
            if len(documents)>= self.k:
                break                

            metadata_value = doc.metadata[self.metadata_parent_field]

            parent_retriever = self.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={'k': self.k, 'filter': {self.metadata_parent_field:metadata_value}},
                        )
            
            docs_part = parent_retriever.get_relevant_documents(query)
            documents.extend(docs_part[:self.k - len(documents)])
 
        return documents
