from sudo import sudo

if __name__ == '__main__':
    sudo('apt-get install python3-pip')
    sudo('pip3 install --upgrade pip -i http://pypi.douban.com/simple --trusted-host pypi.douban.com')
    sudo('pip3 install gevent')
    sudo('pip3 install -i https://pypi.douban.com/simple pycryptodome')
    sudo('pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html')
