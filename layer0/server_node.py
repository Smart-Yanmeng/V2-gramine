from gevent import monkey; monkey.patch_all(thread=False)

from typing import  Callable
import os
from server import SERVER

class SERVERBFTNode (SERVER):

    def __init__(self, id, N, p, t, bft_from_server: Callable, bft_to_client_SERVER: Callable, ip_list, server_bft_mpq):
        self.bft_from_server = bft_from_server#server_get
        self.bft_to_client_SERVER = bft_to_client_SERVER
        self.send_SERVER = lambda j, o: self.bft_to_client_SERVER((j, o))
        self.recv = lambda: self.bft_from_server()
        self.server_bft_mpq = server_bft_mpq
        SERVER.__init__(self, id, N, p, t, self.send_SERVER, self.recv, ip_list, self.server_bft_mpq)
    
    def run(self):

        pid = os.getpid()
        self.run_bft()
        

