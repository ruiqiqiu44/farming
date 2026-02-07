import pyaudiowpatch as pyaudio
import socket
import msvcrt
import threading

host_ip = '127.0.0.1'
port = 65432
chunk = 1024

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True, frames_per_buffer=chunk)#used for play audio
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.connect((host_ip, port))

def check_input():
    if msvcrt.kbhit():  # Returns True if a key is waiting in the buffer
        key = msvcrt.getch().decode('utf-8').lower()
        return key
    return None

def receive_audio():
  try:
    while True:
        key = check_input()
        if key == 'q':
            print("Ending call...")
            break
        audio_data, _ = client_socket.recvfrom(chunk)
        stream.write(audio_data)
  finally:
    client_socket.close()
    
try:
   
  thread = threading.Thread(target=receive_audio)
  thread.start()
finally:
   client_socket.close()