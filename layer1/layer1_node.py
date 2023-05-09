from gevent import monkey; monkey.patch_all(thread=False)

from typing import  Callable
import os
from layer1 import LAYER1

class LAYER1_Node (LAYER1):

    def __init__(self, id, N, p, t, bft_from_server: Callable, bft_to_client_SERVER: Callable, ip_list):
        self.bft_from_server = bft_from_server#server_get
        self.bft_to_client_SERVER = bft_to_client_SERVER
        self.send_SERVER = lambda j, o: self.bft_to_client_SERVER((j, o))#server.put
        self.recv = lambda: self.bft_from_server()
        LAYER1.__init__(self, id, N, p, t, self.send_SERVER, self.recv, ip_list)
    
    def run(self):

        pid = os.getpid()
        self.run_bft()
        



