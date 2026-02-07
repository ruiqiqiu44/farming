import pyaudiowpatch as pyaudio
import socket
import msvcrt
import threading

host_ip = '127.0.0.1'
port = 65432
chunk = 1024
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=chunk)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)






def check_input():
    if msvcrt.kbhit():  # Returns True if a key is waiting in the buffer
        key = msvcrt.getch().decode('utf-8').lower()
        return key
    return None

def send_audio(client_socket):
    
  try:
    while True:
      key = check_input()
      if key == 'q':
        print("Ending call...")
        break
      audio_data = stream.read(chunk) #read stream
      client_socket.send(audio_data)
  finally:
    client_socket.close()


try:
  server_socket.bind((host_ip, port))
  server_socket.listen(1)
  client_socket, address = server_socket.accept()
  thread = threading.Thread(target=send_audio, args=(client_socket,))
  thread.start()
finally:
  server_socket.close()
  