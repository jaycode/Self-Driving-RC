import serial

port = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=3.0)

while True:
    rcv = port.read(8)
    # print(ord(rcv) + "\n")
    print(rcv + "\n")

