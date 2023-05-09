from gevent import monkey; monkey.patch_all(thread=False)

import traceback
from typing import List, Callable
from gevent import Greenlet
from layer4_node import LAYER4_Node
from socket_server import NetworkServer
from socket_client import NetworkClient
from multiprocessing import Value as mpValue, Queue as mpQueue


def instantiate_node(i, N, p, t, bft_from_server: Callable, bft_to_client_SERVER: Callable, ip_list):
	bft = LAYER4_Node(i, N, p, t, bft_from_server, bft_to_client_SERVER, ip_list)
	return bft
	
if __name__ == '__main__':
	
	import argparse
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('id', metavar='id', help='identifier of node', type=int)
	parser.add_argument('N', metavar='N', help='number of all nodes', type=int)
	parser.add_argument('p', metavar='p', help='the number of real layers (nodes)', type=int)
	parser.add_argument('t', metavar='t', help='number of redundent layers', type=int)
	args = parser.parse_args()

	# Some parameters
	i = args.id
	N = args.N
	p = args.p
	t = args.t


    # Nodes list
	addresses_SERVER = [None] * N
	ip_list = [None] * N

	try:
		with open('hosts_SERVER.config', 'r') as hosts:
			for line in hosts:
				params = line.split()
				pid = int(params[0])
				priv_ip = params[1]
				pub_ip = params[2]
				port = int(params[3])
		        # print(pid, ip, port)
				if pid not in range(N):
					continue
				if pid == i:
					my_address_SERVER = (pub_ip, port)
				addresses_SERVER[pid] = (pub_ip, port)
				#ip_list[pid] = pub_ip
				ip_list[pid] = str(port)
		assert all([node is not None for node in addresses_SERVER])
		print("hosts_SERVER.config is correctly read")

       
		client_bft_mpq_SERVER = mpQueue()
		client_from_bft_SERVER = lambda: client_bft_mpq_SERVER.get(timeout=0.00001)
		bft_to_client_SERVER = client_bft_mpq_SERVER.put_nowait

		server_bft_mpq = mpQueue()
		bft_from_server = lambda: server_bft_mpq.get(timeout=0.00001)
		server_to_bft = server_bft_mpq.put_nowait

		netServer = NetworkServer(my_address_SERVER[1], my_address_SERVER[0], i, addresses_SERVER, server_to_bft)
		netClient = NetworkClient(my_address_SERVER[1], my_address_SERVER[0], i, addresses_SERVER, client_from_bft_SERVER)
		bft = instantiate_node(i, N, p, t, bft_from_server, bft_to_client_SERVER, ip_list)

		netServer.start()
		netClient.start()

		print("waiting for network ready...")
		
		bft_thread = Greenlet(bft.run)
		bft_thread.start()
		bft_thread.join()
        
	except FileNotFoundError or AssertionError as e:
	    traceback.print_exc()

