''' 
QdrantBM25Retriever - retriever BM25 based o n Qdrant and Langchain

Library instalation:
pip install qdrant-client     # Qdrant database API
pip install langchain         # langchain framework
'''

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional
from uuid import UUID
from qdrant_client import QdrantClient

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()



# constants
PREPOSITION_SET = {"v", "na", "s", "z", "o", "do", "pro", "k", "za", "po", "při", "od", "podle", "u", "před", 
                    "mezi", "ke", "bez", "proti", "přes", "nad", "ve", "se", }       
CONJUCTION_SET = {"a", "nebo", "aby", "i", "protože", "jinak", "že",  }
WORD_TYPE_SET = PREPOSITION_SET.union(CONJUCTION_SET)



def cs_preprocessing_func(text: str) -> List[str]:
    text_low = text.lower()
            
    # removing extra characters
    translation_table = text_low.maketrans(".,:;?!", "      ")
    word_list = text_low.translate(translation_table).split()

    # Remove words in the set from the list using list comprehension
    word_list = [word for word in word_list if word not in WORD_TYPE_SET]

    return word_list


# Function to get data for a specific id from record list
def get_data_by_id(records_list, id):
    for record in records_list:
        if record.id == id:
            return record


class QdrantBM25Retriever(BaseRetriever):
    """`BM25 retriever with Qdrant."""

    url: str = None
    """ URL of the Qdrant instance to connect to."""

    api_key:str = None
    """ API key to Qdrant """

    collection_name: str = None
    """ Name of the collection to use in Qdrant."""
    
    ids: List[UUID] = []
    """ List of document id's."""
    
    language: str = ""
    """ Language of documents.  cs, if is empty then is used default preprocessing function   """

    client: Any
    """ Qdrant client"""    

    vectorizer: Any
    """ BM25 vectorizer."""
    
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def create(cls,
        url:str,
        api_key:str,
        collection_name:str,   
        language: str = "",
        item_list: List[str] = ["page_content"],
        bm25_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> QdrantBM25Retriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            url - URL of the Qdrant instance to connect to
            api_key: API key to Qdrant
            collection_name: Name of the collection to use in Qdrant
            language: language of documents cs, ...
            item_list: list of metadata, which is used for BM25 retriever (default is ["page_content"])
            bm25_params: Parameters to pass to the BM25 vectorizer.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A QdrantBM25Retriever instance.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )
        
        client = QdrantClient(
            url = url,
            api_key=api_key,
            prefer_grpc=True,               
            )

        result = client.scroll(
            collection_name=collection_name,
            limit=50000,
            with_payload=True,
            with_vectors=False,
        )
        
        records = result[0]
        
        client.close()

        ids = [UUID(item.id) for item in records]

        # text construction
        texts = []
        for item in records:
            text = ""
            for field in item_list:
                if field == "page_content":
                    text += item.payload["page_content"]
                else:
                    if field in item.payload["metadata"]:
                        text += "\n" + item.payload["metadata"][field]

            texts.append(text)

        # texts = [item.payload["page_content"] for item in records]
        
        match language:
            case "cs":
                preprocess_func = cs_preprocessing_func
            case _:
                preprocess_func = default_preprocessing_func


        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)

        return cls(
            url = url, api_key = api_key, collection_name = collection_name,
            ids=ids, language = language, 
            vectorizer=vectorizer, preprocess_func=preprocess_func, **kwargs
        )

    def set_database(self):
        """
        Set a QdrantBM25Retriever from a Qdrant database texts.

        Args:

        Returns:
        """ 

        self.client = QdrantClient(
            url = self.url,
            api_key=self.api_key,
            prefer_grpc=True,               
            )
        
        match self.language:
            case "cs":
                self.preprocess_func = cs_preprocessing_func
            case _:
                self.preprocess_func = default_preprocessing_func 
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        processed_query = self.preprocess_func(query)
        return_ids = self.vectorizer.get_top_n(processed_query, self.ids, n=self.k)
        
        ids = [str(item) for item in return_ids]

        # for ids read texts from Qdrant database
        return_docs = self.client.retrieve(
            collection_name = self.collection_name,
            ids = ids,
            with_payload = True,
            with_vectors = False,            
        )

        documents = []
        for id in ids:
            item = get_data_by_id(return_docs, id)
            documents.append(Document(page_content = item.payload["page_content"], metadata = item.payload["metadata"]))

        return documents