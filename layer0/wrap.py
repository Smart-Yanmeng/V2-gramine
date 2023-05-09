from gevent import monkey;

monkey.patch_all(thread=False)

from utils import strtobytes
from torchvision import datasets, transforms
import struct
from io import BytesIO
import pickle
import torch
import time
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from utils import strtobytes
from gevent import Greenlet


# ========================
# This is the forward wrap
# ========================
def wrap_forward_6():
    message_type = 1
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.seek(0)
    return buf.read()


def wrap_forward_5(random_p_layers, lable, localIP):
    message_type = 1
    filename = "./config/publickey%s.keys" % random_p_layers[5]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_forward_6())))
    buf.write(wrap_forward_6())
    buf.write(struct.pack("<i", len(lable)))
    buf.write(lable)
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(localIP))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_forward_4(random_p_layers, lable, localIP):
    message_type = 1
    filename = "./config/publickey%s.keys" % random_p_layers[4]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_forward_5(random_p_layers, lable, localIP))))
    buf.write(wrap_forward_5(random_p_layers, lable, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[5]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_forward_3(random_p_layers, lable, localIP):
    message_type = 1
    filename = "./config/publickey%s.keys" % random_p_layers[3]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_forward_4(random_p_layers, lable, localIP))))
    buf.write(wrap_forward_4(random_p_layers, lable, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[4]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_forward_2(random_p_layers, lable, localIP):
    message_type = 1
    filename = "./config/publickey%s.keys" % random_p_layers[2]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_forward_3(random_p_layers, lable, localIP))))
    buf.write(wrap_forward_3(random_p_layers, lable, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[3]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_forward_1(random_p_layers, lable, raw_data, localIP):
    message_type = 1
    filename = "./config/publickey%s.keys" % random_p_layers[1]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(raw_data)))
    buf.write(raw_data)
    buf.write(struct.pack("<i", len(wrap_forward_2(random_p_layers, lable, localIP))))
    buf.write(wrap_forward_2(random_p_layers, lable, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[2]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


# -------------------------------
# ---------Test phase------------
# -------------------------------
def wrap_test_forward_6():
    message_type = 3
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.seek(0)
    return buf.read()


def wrap_test_forward_5(random_p_layers, lable, localIP):
    message_type = 3
    filename = "./config/publickey%s.keys" % random_p_layers[5]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_test_forward_6())))
    buf.write(wrap_test_forward_6())
    buf.write(struct.pack("<i", len(lable)))
    buf.write(lable)
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(localIP))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_test_forward_4(random_p_layers, lable, localIP):
    message_type = 3
    filename = "./config/publickey%s.keys" % random_p_layers[4]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_test_forward_5(random_p_layers, lable, localIP))))
    buf.write(wrap_test_forward_5(random_p_layers, lable, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[5]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_test_forward_3(random_p_layers, lable, localIP):
    message_type = 3
    filename = "./config/publickey%s.keys" % random_p_layers[3]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_test_forward_4(random_p_layers, lable, localIP))))
    buf.write(wrap_test_forward_4(random_p_layers, lable, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[4]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_test_forward_2(random_p_layers, lable, localIP):
    message_type = 3
    filename = "./config/publickey%s.keys" % random_p_layers[2]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_test_forward_3(random_p_layers, lable, localIP))))
    buf.write(wrap_test_forward_3(random_p_layers, lable, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[3]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_test_forward_1(random_p_layers, lable, raw_data, localIP):
    message_type = 3
    filename = "./config/publickey%s.keys" % random_p_layers[1]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(raw_data)))
    buf.write(raw_data)
    buf.write(struct.pack("<i", len(wrap_test_forward_2(random_p_layers, lable, localIP))))
    buf.write(wrap_test_forward_2(random_p_layers, lable, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[2]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


# -------------------------------
# ---------Backward phase--------
# -------------------------------

def wrap_backward_6():
    message_type = 2
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.seek(0)
    return buf.read()


def wrap_backward_5(random_p_layers, localIP):
    message_type = 2
    filename = "./config/publickey%s.keys" % random_p_layers[1]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_backward_6())))
    buf.write(wrap_backward_6())
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(localIP))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_backward_4(random_p_layers, localIP):
    message_type = 2
    filename = "./config/publickey%s.keys" % random_p_layers[2]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_backward_5(random_p_layers, localIP))))
    buf.write(wrap_backward_5(random_p_layers, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[1]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_backward_3(random_p_layers, localIP):
    message_type = 2
    filename = "./config/publickey%s.keys" % random_p_layers[3]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_backward_4(random_p_layers, localIP))))
    buf.write(wrap_backward_4(random_p_layers, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[2]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_backward_2(random_p_layers, localIP):
    message_type = 2
    filename = "./config/publickey%s.keys" % random_p_layers[4]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_backward_3(random_p_layers, localIP))))
    buf.write(wrap_backward_3(random_p_layers, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[3]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()


def wrap_backward_1(random_p_layers, localIP):
    message_type = 2
    filename = "./config/publickey%s.keys" % random_p_layers[5]
    RSA_public_key = RSA.importKey(open(filename, 'rb').read())
    buf = BytesIO()
    buf.write(struct.pack("<i", message_type))
    buf.write(struct.pack("<i", len(wrap_backward_2(random_p_layers, localIP))))
    buf.write(wrap_backward_2(random_p_layers, localIP))
    encryptor = PKCS1_OAEP.new(RSA_public_key)
    encrypted = encryptor.encrypt(strtobytes(random_p_layers[4]))
    buf.write(struct.pack("<i", len(encrypted)))
    buf.write(encrypted)
    buf.seek(0)
    return buf.read()
