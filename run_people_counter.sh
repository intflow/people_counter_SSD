
sudo ifconfig eth0 192.168.1.2

python3 people_counter_ssd.py \
	--prototxt models/MobileNetSSD_deploy.prototxt \
	--model models/MobileNetSSD_deploy.caffemodel \
	--confidence 0.3 \
	--skip-frames 5 \
	--resize_frame 160 \
<<<<<<< HEAD
	--uart_port "/dev/ttyAMA0" \
=======
	--uart_port "/dev/ttyUSB0" \
>>>>>>> 7eb1ecc322e19d1e02c42025f4319655ffbf1f3a
	--uart_baud 9600 \
	--input example_01.mp4 \
	--output "" \
<<<<<<< HEAD
	--screen 1

	#--input example_01.mp4 \
=======
	--screen 0

	#--input "rtsp://admin:intflow3121@192.168.1.64:554/Streaming/Channels/102/" \
>>>>>>> 7eb1ecc322e19d1e02c42025f4319655ffbf1f3a
	#--output output/output_01.avi
