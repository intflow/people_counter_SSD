python3 people_counter_ssd.py \
	--prototxt models/MobileNetSSD_deploy.prototxt \
	--model models/MobileNetSSD_deploy.caffemodel \
	--confidence 0.3 \
	--skip-frames 5 \
	--resize_frame 160 \
	--uart_port "/dev/ttyS0" \
	--uart_baud 9600 \
	--input "rtsp://admin:intflow3121@192.168.1.64:554/Streaming/Channels/102/" \
	--output "" \
	--screen 1 
	#--input ../sample4.mp4 \
	#--input videos/example_01.mp4
	#--output output/output_01.avi
