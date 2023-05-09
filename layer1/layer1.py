from gevent import monkey; monkey.patch_all(thread=False)

import struct
from io import BytesIO
import pickle
import torch
import torch.backends.cudnn as cudnn
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from utils import strtobytes, bytestostr
from gevent import Greenlet
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from optparse import OptionParser

global model, optimizer    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def recvall(sock):
    fragments = []
    while True: 
        chunk = sock.recv(100000)
        if not chunk: 
            break
    fragments.append(chunk)
    arr = b''.join(fragments)
    return arr

def unpack_message_forward(m, self):
    global layer_client_message

    filename = "./config/privatekey%s.keys"%localIP
    RSA_private_key =  RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO(m)
    message_type, = struct.unpack("<i", buf.read(4))
    raw_data_length, = struct.unpack("<i", buf.read(4))
    raw_data = pickle.loads(buf.read(raw_data_length))
    computation_data = ML_operation(message_type, raw_data, self)
    message_length, = struct.unpack("<i", buf.read(4))
    message = bytestostr(buf.read(message_length))

    encryped_ip_length, = struct.unpack("<i", buf.read(4))
    encryped_ip = buf.read(encryped_ip_length)
    decryptor = PKCS1_OAEP.new(RSA_private_key)
    next_addr = decryptor.decrypt(encryped_ip)

    '''next_addr_length, = struct.unpack("<i", buf.read(4))
    next_addr = bytestostr(buf.read(next_addr_length))'''

    return next_addr, computation_data, message

def unpack_message_back(m, self):
    buf = BytesIO(m)

    filename = "./config/privatekey%s.keys"%localIP
    RSA_private_key =  RSA.importKey(open(filename, 'rb').read())
    
    zn_length, = struct.unpack("<i", buf.read(4))

    zn = pickle.loads(buf.read(zn_length))
    c_length, = struct.unpack("<i", buf.read(4))
    c = buf.read(c_length)

    buf = BytesIO(c)
    message_type, = struct.unpack("<i", buf.read(4))

    zn_next = ML_operation(message_type, zn, self)
    message_length, = struct.unpack("<i", buf.read(4))
    message = bytestostr(buf.read(message_length))

    encryped_ip_length, = struct.unpack("<i", buf.read(4))
    encryped_ip = buf.read(encryped_ip_length)
    
    decryptor = PKCS1_OAEP.new(RSA_private_key)
    next_addr = decryptor.decrypt(encryped_ip)
    '''next_addr_length, = struct.unpack("<i", buf.read(4))
    next_addr = bytestostr(buf.read(next_addr_length))'''
    return next_addr, zn_next, message

def ML_operation(message_type, message, self):
    global output
    message.to(device)
    '''if message_type == 0:
        model, optimizer = reset_model()
        # return value of y2?
        # y2 can be regarded as a reset signal to client. -Rui
        # Might be redundant for the next layer -Rui
        y2 = "successfully reset model"'''
    
    if message_type == 1:
        # message = message.view(message.shape[0], -1) #Rui
            
        # Cleaning gradients
        self.optimizer.zero_grad()

        # evaluate
        output = self.model(message) #Rui

        #if output.data.size() != (64,64): 
            #continue
        y2 = Variable(output.data, requires_grad=True)
        #last layer - LOSS
        #loss = model(output, labels)
#        print(y2)
    elif message_type == 2:
        #ohter layers:
        output.backward(message) # Rui 
        # optimize the weights
        self.optimizer.step()
        #Prepare y2 for backward sending. - Rui
        y2 = "One iteration finished"
    else:
        # Test or validation. -Rui
        #print(message.shape)
        with torch.no_grad():
            output = self.model(message)
        y2 = output
    return y2

def pack(m1, m2):
    buf = BytesIO()
    m1_pickle = pickle.dumps(m1)
    buf.write(struct.pack("<i", len(m1_pickle)))
    buf.write(m1_pickle)
    buf.write(struct.pack("<i", len(m2)))
    buf.write(strtobytes(m2))
    buf.seek(0)
    return buf.read()

def find(next_addr, self):
    for i in range(self.N):
        if self.ip_list[i] == next_addr:
            return i
            
def get_len(msg, IP):
    buf = BytesIO()
    buf.write(struct.pack("<i", len(msg)))
    buf.write(struct.pack("<i", len(IP)))
    buf.write(IP.encode('utf-8'))
    buf.write(msg)
    buf.seek(0)
    return buf.read()
    
#define models
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

#layer
def reset_model():
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), nn.ReLU())
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #Rui:lr=0.01
    return model, optimizer

class LAYER1():

    def __init__(self, pid, N, p, t, send_SERVER, recv, ip_list):
        global localIP
        global random_server_all_layers
        self.id = pid
        self.N = N
        self.f = p
        self.t = t
        self._send_SERVER = send_SERVER
        self._recv = recv
        self.ip_list = ip_list
        localIP = ip_list[pid]
        self.model, self.optimizer = reset_model()
        if device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
        
    def _broadcast(self, j,o):
        self._send_SERVER(j, o)#client_put

    def run_bft(self):
        """Run the SERVER protocol."""
#        model, optimizer = reset_model()
        def _recv_loop():
            print("Server start, listening to the channel----") 
            flag = True
            while True:
                try:
                    msg = self._recv() #server.get
                    if flag:
                        addr = msg.decode('utf-8')
                        flag = False
                        continue
                    flag = True
#                    print("Address: - - - - - - - - - - - - %s" % addr)
                    if(addr == self.ip_list[0]):
                        next_addr, computation_result, str_message = unpack_message_forward(msg, self)
                        ciphertext = pack(computation_result, str_message)
                        id = find(str(next_addr, encoding="utf-8"), self)
                        val = get_len(ciphertext, localIP)
                        self._broadcast(id, val)
                    else:
                        next_addr, computation_result, str_message = unpack_message_back(msg, self)
                        ciphertext = pack(computation_result, str_message)
                        id = find(str(next_addr, encoding="utf-8"), self)
                        val = get_len(ciphertext, localIP)
                        self._broadcast(id, val)
                except:
                    continue

        self._recv_thread = Greenlet(_recv_loop)
        self._recv_thread.start()

        print('Node %d starts SERVER' % self.id)
        
