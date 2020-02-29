python3 people_counter_ssd.py \
	--prototxt models/MobileNetSSD_deploy.prototxt \
	--model models/MobileNetSSD_deploy.caffemodel \
	--confidence 0.3 \
	--skip-frames 5 \
	--uart_port "/dev/ttyS0" \
	--uart_baud 9600 \
	--input "example_01.mp4" \
	--output "" \
	--screen 0
	#--input ../sample4.mp4 \
	#--input videos/example_01.mp4
	#--output output/output_01.avi
