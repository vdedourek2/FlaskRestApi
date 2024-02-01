''' 
KBAMemory - Class for saving conversation with user (based on Langchain)
'''
import time
from langchain_core.messages import AIMessage, HumanMessage

class KBAMemory(object):
    '''
    Class for saving conversation with user (based on Langchain)
    -----------------------------------------------------------------------
    '''
    k_history:int = 3               # number messages in history
    time_limit_history:int=1200     # garbage collector timeout of the user inactivity in seconds

    history = {}     # chat history {"user_id": { "last_time":last_time, "history":[(question, answer),...]}, ...}
 
    def  _garbage_collector(self):
        """
        Cleaning the history after the time interval
        """
        current_time = time.time()
        users_to_delete = []

        # find out users to dalete
        for user_id, value in self.history.items():
            if current_time - value["last_time"] > self.time_limit_history:
                users_to_delete.append(user_id)

        # deleting inactive users
        for user_id in users_to_delete:
            del self.history[user_id]

        
    def add_history(self,
        user_id: str ="",
        question:str="",
        answer:str="",
    ):
        """
        Add question/answer to history.
        Shrink the history to self.k_history length.
        chat history -> self.history
         {"user_id": { "last_time":last_time, "history":[(question, answer),...]}, ...}
        
        Args:
        user_id - user id
        question - question
        answer - answer
        """
        current_time = time.time()
        user_data = self.history.get(user_id)        

        if user_data:
            user_data["last_time"] = current_time
            user_history = user_data["history"]
            user_history.append((question, answer))
            user_history = user_history[-self.k_history:]   # shrink user_history to k_history items
        else:
            self.history[user_id] = { "last_time":current_time, "history":[(question, answer)]}
    
            
    def get_history(self,
        user_id: str ="",        # user id
    ) -> list:
        """
        Get history for user + Clearing the history after the time interval.
         {"user_id": { "last_time":last_time, "history":[(question, answer),...]}, ...}
        
        Args:
        user_id - user id
        """
        # Clearing the history after the time interval.
        self._garbage_collector()

        chat_history = []
        user_data = self.history.get(user_id)        
  
        if user_data:
            for query, answer in user_data["history"]:
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=answer))
 
        return chat_history

 
    def clear_history(self):
        """
        Clear history
        """
        self.history = []
 
    
    def get_users(self,
        ) -> list:
        '''
        Get list users info from self.history
        '''
        return self.history