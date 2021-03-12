adb -s $PIXEL4 tcpip 5555
adb -s $PIXEL4 connect 192.168.0.44:5555
adb -s $PIXEL5 tcpip 5556
adb -s $PIXEL5 connect 192.168.0.46:5556
export PIXEL4_wifi="192.168.0.44:5555"
export PIXEL5_wifi="192.168.0.46:5556"