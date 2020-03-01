import time
import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='/dev/ttyAMA0',
    baudrate=9600,
    parity='N',
    stopbits=1,
    bytesize=8
)

ser.isOpen()

while 1 :
    # send the character to the device
    # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
    ser.write((bytes('hello\r\n', encoding='ascii')))
    time.sleep(1)
