# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 20:03:00 2022

@author: cbony
"""
import struct
import os
from io import BytesIO
from time import sleep
import hashlib
import sys


# def lite_hash(msg):
#     hasher = hashlib.sha512()
#     hasher.update(msg)
#     return hasher.hexdigest()

# def mylog(*args, **kargs):
#     if not 'verboseLevel' in kargs:
#         kargs['verboseLevel'] = 0
#     if kargs['verboseLevel'] >= verbose:
#         print (" ".join([isinstance(arg, str) and arg or repr(arg) for arg in args]))
#         sys.stdout.flush()
#         sys.stderr.flush()

def deepEncode(msgType, id, payload1, payload2, payload3):
    buf = BytesIO()
    id = str(id)
    buf.write(struct.pack('iiiii', len(msgType), len(id), len(payload1), len(payload2), len(payload3)))
    buf.write(lstrtobytes(msgType))
    buf.write(lstrtobytes(id))
    buf.write(strtobytes(payload1))
    buf.write(strtobytes(payload2))
    buf.write(strtobytes(payload3))
    buf.seek(0)
    return buf.read()


# def bytestostr(m):
#     return m.decode("ISO-8859-1")

# def strtobytes(m):
#     return m.encode("ISO-8859-1")

def lstrtobytes(m):
    return m.encode("utf-8")


# def lbytestostr(m):
#     return m.decode("utf-8")

def bytestostr(m):
    return m.decode("ISO-8859-1")


def strtobytes(m):
    return m.encode("ISO-8859-1")


def pad_message(message):
    """
    Pads a string for use with AES encryption
    :param message: string to be padded
    :return: padded message
    """
    pad_size = 16 - (len(message) % 16)
    if pad_size == 0:
        pad_size = 16
    message += chr(pad_size) * pad_size
    return message
