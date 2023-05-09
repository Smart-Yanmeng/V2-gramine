from gevent import monkey; monkey.patch_all(thread=False)

from gevent.server import StreamServer
import pickle
from typing import Callable
import os
import logging
import traceback
from multiprocessing import Value as mpValue, Process
import time
import struct
from io import BytesIO
# Network node class: deal with socket communications
class NetworkServer (Process):

    #SEP = '\r\nSEP\r\nSEP\r\nSEP\r\n'.encode('utf-8')

    def __init__(self, port: int, my_ip: str, id: int, addresses_list: list, server_to_bft: Callable):

        self.server_to_bft = server_to_bft
        self.ip = "0.0.0.0"
        self.port = port
        self.id = id
        self.addresses_list = addresses_list
        self.N = len(self.addresses_list)
        self.is_in_sock_connected = [False] * self.N
        self.socks = [None for _ in self.addresses_list]
        super().__init__()
        
    def _listen_and_recv_forever(self):
        print("my IP is " + self.ip + " my port is " + str(self.port))
        def _handler(sock, address):
            #buf = b''
            tmp = b''
            try:
                while True:
                    tmp += sock.recv(200000)
                    
                    buf = BytesIO(tmp)
                    size, = struct.unpack("<i",buf.read(4))
                    ip_size, = struct.unpack("<i",buf.read(4))
                    if len(tmp) - 8 - ip_size != size:
                        continue
                    IP = buf.read(ip_size)
                    self.server_to_bft(IP)#sever_put
                    msg = buf.read(size)
                    self.server_to_bft(msg)#sever_put
                    tmp = b''
            except Exception as e:
                print(str((e, traceback.print_exc())))
                
        self.streamServer = StreamServer((self.ip, self.port), _handler)
        self.streamServer.serve_forever()


    def run(self):
        pid = os.getpid()
        self._listen_and_recv_forever()

    def _address_to_id(self, address: tuple):
        for i in range(self.N):
            if address[0] != '127.0.0.1' and address[0] == self.addresses_list[i][0]:
                return i
        return int((address[1] - 10000) / 200)

