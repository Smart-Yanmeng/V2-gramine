from gevent import monkey; monkey.patch_all(thread=False)

import struct
from io import BytesIO
from utils import strtobytes, bytestostr
import pickle
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import struct
from io import BytesIO
from utils import strtobytes, bytestostr
import torch.nn as nn
from gevent import Greenlet
from optparse import OptionParser
import torch
import torch.backends.cudnn as cudnn

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

def unpack_message_back(m, self):
    buf = BytesIO(m)

    filename = "./config/privatekey%s.keys"%localIP
    RSA_private_key =  RSA.importKey(open(filename, 'rb').read())

    message_type, = struct.unpack("<i", buf.read(4))
    computation_data = ML_operation(self, message_type, message=0, labels=0)
    message_length, = struct.unpack("<i", buf.read(4))
    message = bytestostr(buf.read(message_length))
    encryped_ip_length, = struct.unpack("<i", buf.read(4))
    encryped_ip = buf.read(encryped_ip_length)
    decryptor = PKCS1_OAEP.new(RSA_private_key)
    next_addr = decryptor.decrypt(encryped_ip)
    '''next_addr_length, = struct.unpack("<i", buf.read(4))
    next_addr = bytestostr(buf.read(next_addr_length))'''
    return next_addr, computation_data, message

def ML_operation(self, message_type, message, labels=0): #Rui
    #return b"haha5"
    global output
    '''if message_type == 0:
        model, optimizer = reset_model()
        # return value of y2?
        # y2 can be regarded as a reset signal to client. -Rui
        # Might be redundant for the next layer -Rui
        y2 = "successfully reset model"'''
    if message_type == 1:
        message.to(device)
        # message = message.view(message.shape[0], -1) #Rui
            
        # Cleaning gradients
        # optimizer.zero_grad()
        # evaluate
        output = self.model(message, labels) #Rui
        # The labels is randomly generated, see client.py, labels, need to solve double for-loops then we can use correct labels from data set. -Rui 
        #if output.data.size() != (64,64): 
            #continue
        # last layer
        y2 = output
    elif message_type == 2:
        #print("output 2---", output.item())
        #loss:
        #To chao: store message from layer4
        output.backward()
        # optimizer.step()
        # Prepare y2 for backward sending. -Rui
        y2 = layer4_message.grad
    else:
        y2 = message
    return y2

# To chao: check it
def pack(m1,m2):
    buf = BytesIO()
    m1_pickle = pickle.dumps(m1)
    buf.write(struct.pack("<i", len(m1_pickle)))
    buf.write(m1_pickle)
    buf.write(struct.pack("<i", len(m2)))
    buf.write(strtobytes(m2))
    buf.seek(0)
    return buf.read()

def unpack_message_forward(m, self):
    global layer4_message

    filename = "./config/privatekey%s.keys"%localIP
    RSA_private_key =  RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO(m)
    zn_length, = struct.unpack("<i", buf.read(4))
    
    zn = pickle.loads(buf.read(zn_length))
    layer4_message = zn
    c_length, = struct.unpack("<i", buf.read(4))
    c = buf.read(c_length)
    buf = BytesIO(c)
    message_type, = struct.unpack("<i", buf.read(4))
    message_length, = struct.unpack("<i", buf.read(4))
    message = bytestostr(buf.read(message_length))
    lable_message_length, = struct.unpack("<i", buf.read(4))
    lable_message = pickle.loads(buf.read(lable_message_length))
    encryped_ip_length, = struct.unpack("<i", buf.read(4))
    encryped_ip = buf.read(encryped_ip_length)

    decryptor = PKCS1_OAEP.new(RSA_private_key)
    next_addr = decryptor.decrypt(encryped_ip)
    '''next_addr_length, = struct.unpack("<i", buf.read(4))
    next_addr = bytestostr(buf.read(next_addr_length))'''
   
    # zn: output from layer4, message: labels
    loss = ML_operation(self, message_type, zn, lable_message)
    #print(output.shape)
    # return next_addr, zn_next, message
    # To chao: check it.
    #return next_addr, loss
    return next_addr, loss, message
    
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
    
#layer
def reset_model():
    model = nn.NLLLoss()
    model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    # return model, optimizer
    return model

class LAYER5():

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
        # random_server_all_layers = addresses_SERVER
        random_server_all_layers = self.ip_list
        localIP = random_server_all_layers[pid]
        self.model = reset_model()
        if device == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True

    def _broadcast(self, j, o):
        self._send_SERVER(j, o)#client_put

    def run_bft(self):
        """Run the SERVER protocol."""

        def _recv_loop():
            """Receive messages."""
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
                    if(addr == self.ip_list[0]):
                        next_addr, computation_result, str_message = unpack_message_back(msg, self)
                        ciphertext = pack(computation_result, str_message)
                        id = find(str(next_addr, encoding="utf-8"), self)
                        val = get_len(ciphertext, localIP)
                        self._broadcast(id, val)
                    else:
                        next_addr, computation_result, message  = unpack_message_forward(msg, self)
                        ciphertext = pack(computation_result, message)
                        id = find(str(next_addr, encoding="utf-8"), self)
                        val = get_len(ciphertext, localIP)
                        self._broadcast(id, val)
                except:
                    continue

        self._recv_thread = Greenlet(_recv_loop)
        self._recv_thread.start()

        print('Node %d starts SERVER' % self.id)