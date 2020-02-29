python people_counter_ssd_pi.py \
	--prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
	--confidence 0.4 \
	--skip-frames 4 \
	--input videos/example_01.mp4 
    --output output/example_01.avi
	#--input ../sample4.mp4 \
	#--input videos/example_01.mp4
	#--output output/output_01.avi
