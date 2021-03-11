from urllib.parse import urlparse

import signal
import select
import socket
import threading
import traceback
import time
import sys


# Changing the buffer_size and delay, you can improve the speed and bandwidth.
# But when buffer get to high or delay go too down, you can broke things
buffer_size = 128*4096
delay = 0.0001

class Forward:
    def __init__(self):
        self.forward = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self, host, port):
        try:
            self.forward.connect((host, port))
            return self.forward
        except:
            traceback.print_exc()
            return False


def parse_address(address):
    if '//' not in address:
      address = '//' + address
    address = urlparse(address).netloc
    assert ':' in address
    host, port = address.split(':')
    port = int(port)
    return host, port

class ProxyServer(threading.Thread):
  def __init__(self, remote_address, local_address='localhost:2242'):
    super().__init__(daemon=True)
    self.local_address = local_address
    self.remote_address = remote_address
    print('ProxyServer(remote_address={!r}, local_address={!r})'.format(remote_address, local_address))
    
    # Shutdown on Ctrl+C
    signal.signal(signal.SIGINT, self.shutdown) 

    # Create a TCP socket
    self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Re-use the socket
    self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # bind the socket to a public host, and a port   
    host, port = parse_address(self.local_address)
    self.server.bind((host, port))
    
    self.server.listen(200) # become a server socket
    self.channel = {}
    self.input_list = set()
    self.clients = set()
    self.done = False
    self.prompt = None

  def is_quiet(self):
    return bool(int(os.environ.get('QUIET', '0')))

  def write(self, data):
    if isinstance(data, str):
      sys.stdout.write(data)
    else:
      sys.stdout.buffer.write(data)

  def flush(self):
    sys.stdout.buffer.flush()

  def shutdown(self, *args):
    self.done = True
    self.server.close()
    sys.exit(0)

  def run(self):
      self.input_list.add(self.server)
      while not self.done:
          time.sleep(delay)
          inputready, outputready, exceptready = select.select(self.input_list, [], [])
          for self.s in inputready:
              if self.s == self.server:
                  self.on_accept()
                  break

              self.data = self.s.recv(buffer_size)
              if len(self.data) == 0:
                  self.on_close()
                  break
              else:
                  self.on_recv()
    
  def on_accept(self):
      forward_to = parse_address(self.remote_address)
      forward = Forward().start(forward_to[0], forward_to[1])
      clientsock, clientaddr = self.server.accept()
      if forward:
          print(clientaddr, "has connected")
          self.input_list.add(clientsock)
          self.input_list.add(forward)
          self.clients.add(clientsock)
          self.channel[clientsock] = forward
          self.channel[forward] = clientsock
      else:
          print("Can't establish connection with remote server.")
          print("Closing connection with client side", clientaddr)
          clientsock.close()

  def on_close(self):
      print(self.s.getpeername(), "has disconnected")
      #remove objects from input_list
      self.input_list.remove(self.s)
      self.input_list.remove(self.channel[self.s])
      if self.s in self.clients:
        self.clients.remove(self.s)
      if self.channel[self.s] in self.clients:
        self.clients.remove(self.channel[self.s])
      out = self.channel[self.s]
      # close the connection with client
      self.channel[out].close()  # equivalent to do self.s.close()
      # close the connection with remote server
      self.channel[self.s].close()
      # delete both objects from channel dict
      del self.channel[out]
      del self.channel[self.s]

  def on_recv(self):
      data = self.data
      if self.s in self.clients:
        prompt = b'>>>'
      else:
        prompt = b'---<<<'
      if prompt != self.prompt:
        self.write(b'\n'+prompt)
        self.flush()
        self.prompt = prompt
      # here we can parse and/or modify the data before send forward
      #print(data)
      if self.is_quiet():
        self.write('{} bytes\n'.format(len(data)))
      else:
        self.write(data)
      #sys.stdout.buffer.flush()
      self.channel[self.s].send(data)
    

  # def run(self):
  #   while True:
  #     # Establish the connection
  #     (clientSocket, client_address) = self.serverSocket.accept() 
      
  #     d = threading.Thread(name=self._getClientName(client_address), 
  #     target = self.proxy_thread, args=(clientSocket, client_address))
  #     d.setDaemon(True)
  #     d.start()

  # def _getClientName(self, address):
  #   return 'client'

  # def proxy_thread(self, client_socket, client_address):
  #   pass


if __name__ == '__main__':
    import sys
    import os
    args = sys.argv[1:]
    remote = args[0]
    remote_host, remote_port = parse_address(remote)
    host = os.environ.get('HOST', 'localhost')
    port = int(os.environ.get('PORT', str(remote_port)))
    server = ProxyServer(remote_address=remote, local_address='{}:{}'.format(host, port))
    try:
        server.run()
    except KeyboardInterrupt:
        print("Ctrl C - Stopping server")
        sys.exit(1)
