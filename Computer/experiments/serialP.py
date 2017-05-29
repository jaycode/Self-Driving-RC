import serial

port = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=3.0)

if True:
    print("good")
    port.write("10\n")
    rcv = port.read(10)
    print(rcv)
    #port.write("\r\nYou sent:" + repr(rcv))
