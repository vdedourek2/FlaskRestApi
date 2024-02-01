
''' 
PgvectorClient - Class for PostgreSQL vector database by Langchain

Library instalation:
'''

from dotenv import load_dotenv      # python-dotenv
from dataclasses import dataclass
from qdrant_client.conversions import common_types as types
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from Processing.db_mod import KBADatabase

@dataclass
class PgvectorClient:
    """Class for PostgreSQL vector database by Langchain."""
    db:KBADatabase = None

    def __init__(self,
        ):
        load_dotenv()
        self.db = KBADatabase()


    def get_collection(self,
        collection_name:str,
        ) -> dict:
        '''
        Get information about collection
        Args:
            collection_name - collection name
        '''
        sql = 'SELECT name FROM public.langchain_pg_collection where name = :collection_name;'
        data = {"collection_name":collection_name}
        
        name = self.db.read_value(sql, data)
        if name == None:
            raise Exception(f"Collection '{collection_name}' doesn't exist")
        else:
            return {"collection": name}

    def get_collections(self,
        ) -> list[str]:
        '''
        Get collection list from vector database
        Args:
            collection_name - collection name
        '''
        sql = 'SELECT name FROM public.langchain_pg_collection;'
        data = {}
        
        result = self.db.read_values(sql, 5000, data)
        if result == None:
            return []
        else:
            return [record[0] for record in result]


    def scroll(self,
            collection_name:str,
            limit:int=1000,
            with_payload:bool=True,
            with_vectors:bool=False,
        ) -> List[types.Record]:
        
        rows_result = []

        SQL = (
            "SELECT emb.custom_id, emb.document, emb.cmetadata "
            "FROM public.langchain_pg_embedding as emb "
            "JOIN public.langchain_pg_collection as col on "
	        "    emb.collection_id = col.uuid "
            "WHERE col.name = :collection_name;"
        )        

        data = {"collection_name":collection_name}
        
        result = self.db.read_values(SQL, limit, data)
        if result != None:
            for row in result:
                record = types.Record(id=row[0], payload = {"page_content":row[1], "metadata":row[2]})
                rows_result.append(record)
 
        return rows_result