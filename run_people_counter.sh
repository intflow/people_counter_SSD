
sudo ifconfig eth0 192.168.1.2 up

python3 people_counter_ssd.py \
	--prototxt models/MobileNetSSD_deploy.prototxt \
	--model models/MobileNetSSD_deploy.caffemodel \
	--confidence 0.02 \
	--skip-frames 3 \
	--resize_frame 80 \
	--uart_port "/dev/ttyAMA0" \
	--uart_baud 9600 \
	--input "rtsp://admin:intflow3121@192.168.1.64:554/Streaming/Channels/102/" \
	--output "" \
	--screen 0

	#--input example_01.mp4 \
	#--screen 0
