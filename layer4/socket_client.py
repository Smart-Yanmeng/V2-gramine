from gevent import monkey; monkey.patch_all(thread=False)

import time
import pickle
from typing import List, Callable
import gevent
import os
from multiprocessing import Value as mpValue, Process
from gevent import socket, lock
from gevent.queue import Queue
import logging
import traceback
from ctypes import c_bool

# Network node class: deal with socket communications
class NetworkClient (Process):

    SEP = '\r\nSEP\r\nSEP\r\nSEP\r\n'.encode('utf-8')

    def __init__(self, port: int, my_ip: str, id: int, addresses_list: list, client_from_bft: Callable):

        self.client_from_bft = client_from_bft
       
        self.ip = my_ip
        self.port = port
        self.id = id
        self.addresses_list = addresses_list
        self.N = len(self.addresses_list)

        self.is_out_sock_connected = [False] * self.N

        self.socks = [None for _ in self.addresses_list]
        self.sock_queues = [Queue() for _ in self.addresses_list]
        self.sock_locks = [lock.Semaphore() for _ in self.addresses_list]

        super().__init__()


    def _connect_and_send_forever(self):
        pid = os.getpid()
        while True:
            try:
                if not self.is_out_sock_connected[self.id + 1]:
                    self.is_out_sock_connected[self.id + 1] = self._connect(self.id + 1)
                if not self.is_out_sock_connected[self.id - 1]:
                    self.is_out_sock_connected[self.id - 1] = self._connect(self.id - 1)
                if self.is_out_sock_connected[self.id + 1] and self.is_out_sock_connected[self.id - 1]:
                    break
            except Exception as e:
                print(str((e, traceback.print_exc())))
        send_threads = [gevent.spawn(self._send, j) for j in range(self.N)]
        self._handle_send_loop()


    def _connect(self, j: int):
        sock = socket.socket()
        try:
            sock.connect(self.addresses_list[j])
            self.socks[j] = sock
            return True
        except Exception as e1:
            return False

    def _send(self, j:int):
        while True:
            o = self.sock_queues[j].get()
            try:
                self.socks[j].sendall(o)
            except:
                self.socks[j].close()
                break


    ##
    def _handle_send_loop(self):
        while True:
            try:
                j, o = self.client_from_bft()#client_get
                try:
                    if j == -1: # -1 means broadcast
                        for i in range(self.N):
                            self.sock_queues[i].put_nowait(o)
                    else:
                        self.sock_queues[j].put_nowait(o)
                except Exception as e:
                    print(str(("problem objective when sending", o)))
                    traceback.print_exc()
            except:
                pass



    def run(self):
        pid = os.getpid()
        self._connect_and_send_forever()

