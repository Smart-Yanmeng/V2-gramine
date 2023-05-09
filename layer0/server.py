from gevent import monkey;

monkey.patch_all(thread=False)

from utils import strtobytes
from torchvision import datasets, transforms
import struct
from io import BytesIO
import pickle
import torch
import torch.backends.cudnn as cudnn
import time
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from utils import strtobytes
from gevent import Greenlet
from wrap import wrap_backward_1, wrap_test_forward_1, wrap_forward_1

EPOCH = 10
global random_server_all_layers
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_public_and_private_key(all_layers, localIP):
    keyPair = RSA.generate(1024)

    f_client_publickey = open("./config/publickey%s.keys" % localIP, "wb")
    f_client_privatekey = open("./config/privatekey%s.keys" % localIP, "wb")
    f_client_publickey.write(keyPair.publickey().exportKey())
    f_client_privatekey.write(keyPair.exportKey())
    f_client_publickey.close()
    f_client_privatekey.close()

    for i in range(len(all_layers)):
        keyPair = RSA.generate(1024)
        f1 = open("./config/publickey%s.keys" % all_layers[i], "wb")
        f2 = open("./config/privatekey%s.keys" % all_layers[i], "wb")
        f1.write(keyPair.publickey().exportKey())
        f2.write(keyPair.exportKey())
        f1.close()
        f2.close()


class SERVER():

    def __init__(self, pid, N, p, t, send_SERVER, recv, ip_list, server_bft_mpq):
        global localIP
        self.id = pid
        self.N = N
        self.f = p
        self.t = t
        self._send_SERVER = send_SERVER
        self._recv = recv
        self.ip_list = ip_list
        localIP = ip_list[pid]
        self.server_bft_mpq = self.server_bft_mpq

    def get_len(self, msg, IP):
        buf = BytesIO()
        buf.write(struct.pack("<i", len(msg)))
        buf.write(struct.pack("<i", len(IP)))
        buf.write(IP.encode('utf-8'))
        buf.write(msg)
        buf.seek(0)
        return buf.read()

    def broadcast(self, j, o):
        self._send_SERVER(j, o)

    def get_loss(self):
        stop_ind = 0
        loss_re = 0
        flag = True
        while True:
            if stop_ind:
                break
            try:
                msg = self._recv()  # server.get
                if flag:
                    addr = msg.decode('utf-8')
                    flag = False
                    continue
                flag = True
                buf = BytesIO(msg)
                loss_length, = struct.unpack("<i", buf.read(4))
                loss = pickle.loads((buf.read(loss_length)))
                c_length, = struct.unpack("<i", buf.read(4))
                c = buf.read(c_length)
                buf = BytesIO(c)
                message_type, = struct.unpack("<i", buf.read(4))
                if (message_type == 3):
                    return loss
                if (loss != "One iteration finished"):
                    m = wrap_backward_1(self.ip_list, localIP)
                    val = self.get_len(m, localIP)
                    self.broadcast(5, val)
                    loss_re = loss
                else:
                    stop_ind = 1
                    return loss_re
            except:
                continue

    def run_bft(self):
        """Run the SERVER protocol."""
        # generate_public_and_private_key(random_server_all_layers, localIP)
        """Receive messages."""
        print("client start...")
        random_server_all_layers = self.ip_list
        global localIP
        localIP = random_server_all_layers[self.id]

        # Data preparation
        # Training and validation data are downloaded to '/' 
        # define transformations
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
        trainset = datasets.MNIST('./config/data', download=True, train=True, transform=transform)
        valset = datasets.MNIST('./config/lable', download=True, train=False, transform=transform)

        # Dataloader preparation
        ind = list(range(0, len(trainset), 1))  # every 1st one, take a sample 
        trainset = torch.utils.data.Subset(trainset, ind)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)  #
        valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)  #

        # =====================================
        # ===========Training phase============
        # =====================================

        time1 = time.time()
        for e in range(EPOCH):
            running_loss = 0
            img = 0
            for images, labels in trainloader:
                # Flatten MNIST images into a 784 long vector
                images = images.to(device)
                labels = labels.to(device)
                images = images.view(images.shape[0], -1)

                raw_data = pickle.dumps(images)

                labels_messages = pickle.dumps(labels)

                # timepack = time.time()
                m = wrap_forward_1(random_server_all_layers, labels_messages, raw_data, localIP)
                # timepack2 = time.time()
                # print("message pack time--", timepack2 - timepack)
                val = self.get_len(m, localIP)
                self.broadcast(1, val)
                loss = self.get_loss()
                running_loss += loss.item()
                # print("Epoch {} {} - Training loss: {}".format(e, img, running_loss/len(trainloader)))
                print("Epoch {} {} - Training loss one batch: {} - Training loss average over the loader: {}".format(e,
                                                                                                                     img,
                                                                                                                     loss,
                                                                                                                     running_loss / len(
                                                                                                                         trainloader)))
                img = img + 1
                t = time.time()
                # with open('./data/train.txt', 'a') as f:
                #     s = str(e) + " " + str(img) + " " + str(loss) + " " + str((running_loss/len(trainloader))) + " " + str((t - time1))
                #     f.write(s + '\n')

        print("Training End")
        time2 = time.time()

        # =====================================
        # ===========Testing start============
        # =====================================
        with open('./data/test.txt', 'a') as f:
            f.write("EPOCH " + "ACCURARY" + '\n')

        correct_count, all_count = 0, 0
        for images, labels in valloader:
            images = images.view(images.shape[0], -1)
            # Prepare test data to layer1. -Rui 
            raw_data = pickle.dumps(images)
            loss_message = pickle.dumps(None)  # Don't transmit labels in test phase. -Rui

            m = wrap_test_forward_1(random_server_all_layers, loss_message, raw_data, localIP)

            # connet_and_send_message(random_server_all_layers[0], BASE_PORT,  m)
            val = self.get_len(m, localIP)
            self.broadcast(1, val)

            # Forward model output, fitted to accuracy calculation. -Rui
            Testback = self.get_loss()

            ps = torch.exp(Testback)
            pred_label_tensor = torch.max(ps, dim=1)[1]
            pred_label = pred_label_tensor.detach().numpy()
            true_label = labels.numpy()

            correct_count_current = sum((pred_label == true_label).astype(int))
            correct_count = correct_count + correct_count_current

            all_count += 64

            print("Current epoch {} accumulated accuracy {}".format(all_count / 64, correct_count / all_count))
            # with open('./data/text.txt', 'a') as f:
            #     s = str((all_count / 64)) + " " + str((correct_count/all_count))
            #     f.write(s + '\n')

        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (correct_count / all_count))
        time3 = time.time()
        print("training time = ", time2 - time1)
        print("testing time = ", time3 - time2)
