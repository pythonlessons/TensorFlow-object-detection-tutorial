# TensorFlow CSGO custom object detection faster aim bot
### Original text version of tutorial you can visit [here](http://pylessons.com/).

Welcome to part 8 of the TensorFlow Object Detection API tutorial series. In this I will show you how to export newly trained model and we'll test it out.

So in previous tutorial we made a final working model, chich shoots enemies, but our FPS were really slow, so I decided to try training another model, so that's what we will talk about in this tutorial.

I used almost all the same files from 5th tutorial part, so if you don't have them yet you can clone my GitHub repository. In this part I am not covering how to label pictures, generate tfrecord or configure your training files. I already did this on my 5th tutorial. In this tutorial I will cover only this, which were not covered before.

At first trained model in 5th tutorial I used faster_rcnn_inception_v2_coco model, now I decided to train ssdlite_mobilenet_v2_coco, this model detects objects 21% worse but it is 53% faster, so I decided give it a try. Here is the [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) of all models, so download one if you decided to train model by yourself.

This time when I tries to use train.py file it said that I am using wrong training method, and offered to use model_main.py file. So I uploaded it if someone has problems finding it. I faced some problems when I tried to start it training the model, there were some error but I didn't made a note about them so I can't tell it exactly, so if you face errors, write it on youtube comments, we'll try to solve it.

When training new model I was using same file structure, so you will need only to update ssdlite_mobilenet_v2_coco.config file and download your pretrained model. From TensorFlow/research/object_detection folder continue with following line in cmd:
```
python model_main.py --alsologtostderr --model_dir=CSGO_training_dir/ --pipeline_config_path=CSGO_training/ssdlite_mobilenet_v2_coco.config
```

When training model, it will not show steps as it was doing in 5th tutorial, but training routine periodically saves model checkpoints about every ten minutes to CSGO_training_dir directory. So you should check how your training is going in TensorFlow tensorboard, you can do so with following command:
```
C:\TensorFlow\research\object_detection>tensorboard --logdir=CSGO_training_dir
```

I was training my model since I saw that my loss curve stopped rising. It took for me close to 24 hours and did around 21k training steps.

Then I used same export_inference_graph.py as we used in 6th tutorial. From command promt issued the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path CSGO_training/ssdlite_mobilenet_v2_coco.config --trained_checkpoint_prefix CSGO_training_dir/model.ckpt-XXXX --output_directory CSGO_inference_graph
```

In final step, we tooks all files from my 7th tutorial and replaced CSGO_frozen_inference_graph.pb file with newly trained inference_graph.

Next we tried to play CS:GO and I let my bot to shoot enemies, you can check this out on my YouTube video.

That’s all for this tutorial. With new model it didn't solved FPS problem, it improved sligtly but not that we could play our game. So for future work I decided to learn doing stuff on multiprocessing and run our code processes in parallel. So in next tutorial I will doing stuff with multiprocessing.

### Original text version of tutorial you can visit [here](http://pylessons.com/).
