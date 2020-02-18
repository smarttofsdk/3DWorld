import socket
import win32api
import win32con
import re

ENABLE_SOCKET = False
ENABLE_VIEW = True
ENABLE_ZMQ = False
ENABLE_JS = False

if ENABLE_SOCKET:
    # 报文
    open = bytearray([0x42, 0x54, 0x47, 0x57, 0x04, 0x00, 0x02, 0x01, 0x01, 0x3C])
    close = bytearray([0x42, 0x54, 0x47, 0x57, 0x04, 0x00, 0x02, 0x01, 0x00, 0x3B])
    up = bytearray([0x42, 0x54, 0x47, 0x57, 0x04, 0x00, 0x02, 0x01, 0x02, 0x3D])
    down = bytearray([0x42, 0x54, 0x47, 0x57, 0x04, 0x00, 0x02, 0x01, 0x03, 0x3E])
    # IP和端口
    IP = '139.196.197.176'
    PORT = 10555
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((IP, PORT))

if ENABLE_JS:
    def pre_up():
        win32api.keybd_event(38, 0, 0, 0)
        win32api.keybd_event(38, 0, win32con.KEYEVENTF_KEYUP, 0)

    def pre_down():
        win32api.keybd_event(40, 0, 0, 0)
        win32api.keybd_event(40, 0, win32con.KEYEVENTF_KEYUP, 0)

    def pre_left():
        win32api.keybd_event(37, 0, 0, 0)
        win32api.keybd_event(37, 0, win32con.KEYEVENTF_KEYUP, 0)

    def pre_right():
        win32api.keybd_event(39, 0, 0, 0)
        win32api.keybd_event(39, 0, win32con.KEYEVENTF_KEYUP, 0)

    def pre_enter():
        win32api.keybd_event(13, 0, 0, 0)
        win32api.keybd_event(13, 0, win32con.KEYEVENTF_KEYUP, 0)

    def pre_stop():
        win32api.keybd_event(32, 0, 0, 0)
        win32api.keybd_event(32, 0, win32con.KEYEVENTF_KEYUP, 0)

    def pre_addV():
        win32api.keybd_event(107, 0, 0, 0)
        win32api.keybd_event(107, 0, win32con.KEYEVENTF_KEYUP, 0)     

    def pre_reduceV():
        win32api.keybd_event(109, 0, 0, 0)
        win32api.keybd_event(109, 0, win32con.KEYEVENTF_KEYUP, 0)

        