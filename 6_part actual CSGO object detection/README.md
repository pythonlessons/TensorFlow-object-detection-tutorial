# TensorFlow Object Detection CSGO actual object detection
### CSGO_frozen_inference_graph.pb download [link](https://drive.google.com/open?id=1U6JBcTKPEG9pxviCidVhkPe459XSJlXm).

Welcome to part 6 of our TensorFlow Object Detection API tutorial series. In this part, we're going to export inference graph and detect our own custom objects.

<div align="center">
  <a href="https://www.youtube.com/watch?v=pXyATW0h3zE" target="_blank"><img src="https://github.com/pythonlessons/TensorFlow-object-detection-tutorial/blob/master/1_part%20images/6_YouTube.JPG" alt="TensorFlow object detection tutorial"></a>
</div>

Now when training is complete, the last step is to generate the frozen inference graph (our detection model). From ```/object_detection``` folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path CSGO_training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix CSGO_training/model.ckpt-XXXX --output_directory CSGO_inference_graph
```

This creates a frozen_inference_graph.pb file in the /object_detection/CSGO_inference_graph folder. The .pb file contains the object detection classifier. Rename it to CSGO_frozen_inference_graph.pb and move it to your python folder.
