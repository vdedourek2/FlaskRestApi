''' 
KBAAgent - Class for conversation RAG agent
'''
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

from enum import Enum, auto

class Xtools(Enum):
    RAG = auto()
    SEARCH = auto()
    TEXTQA = auto()


class KBAAgent(object):
    '''
    Class for saving conversation with user (based on Langchain)
    -----------------------------------------------------------------------
    '''
    k_history:int = 3               # number messages in history
    time_limit_history:int=1200     # garbage collector timeout of the user inactivity in seconds

    history = {}     # chat history {"user_id": { "last_time":last_time, "history":[(question, answer),...]}, ...}

    def __init__(self,
        llm:None,


    ):
        pass

 
    def  invoke(self,
        query:str,                
                ) -> str:
        """
        Get answer to question
        """
        pass

