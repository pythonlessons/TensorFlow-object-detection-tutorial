# TensorFlow Object Detection CSGO actual object detection
### CSGO_frozen_inference_graph.pb download [link](https://drive.google.com/open?id=1U6JBcTKPEG9pxviCidVhkPe459XSJlXm).

Welcome to part 6 of our TensorFlow Object Detection API tutorial series. In this part, we're going to export inference graph and detect our own custom objects.

<div align="center">
  <a href="https://www.youtube.com/watch?v=pXyATW0h3zE" target="_blank"><img src="https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/blob/master/1_part%20images/6_YouTube.JPG" alt="TensorFlow object detection tutorial"></a>
</div>

#### Export Inference Graph:

Now when training is complete, the last step is to generate the frozen inference graph (our detection model). Copy export_inference_graph.py file and paste it to ```/object_detection``` folder, then issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path CSGO_training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix CSGO_training/model.ckpt-XXXX --output_directory CSGO_inference_graph
```

### Use our trained custom object detection classifier:

This creates a frozen_inference_graph.pb file in the /object_detection/CSGO_inference_graph folder. The .pb file contains the object detection classifier. Rename it to CSGO_frozen_inference_graph.pb and move it to your main working folder. Also take same labelmap file as you used for training, in my case I renamed it to CSGO_labelmap.pbtxt. Then I took ```object_detection_tutorial_grabscreen_faster.py``` my code from my own 4th tutorial and renamed it to CSGO_object_detection and changed few lines, that it could work for us:

Changed line 39 to my frozen inference graph file.
<br>```PATH_TO_FROZEN_GRAPH = 'CSGO_frozen_inference_graph.pb'```

Changed line 41 to my labelmap file.
<br>```PATH_TO_LABELS = 'CSGO_labelmap.pbtxt'```

And lastly before running the Python scripts, you need to modify the line 42 NUM_CLASSES variable in the script to equal the number of classes we want to detect. I am using only 4 classes, so I changed it to 4:
<br>```NUM_CLASSES = 4```

If everything is working properly, the object detector will initialize for about 10 (for GPU may take a little longer) seconds and then display a custom window size showing objects it’s detected in the image, in our case it's detecting players in CSGO game.

