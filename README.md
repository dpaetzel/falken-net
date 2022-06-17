# falken-net


## Goal


Be able to distinguish between 4 states on [the falcon live
feed](https://start.video-stream-hosting.de/player.html?serverip=116.202.235.106&serverapp=wsgs-live&streamname=Falken.smil).


The states we want to distinguish are:

- Tom and Tina are both there
- only Tom is there
- only Tina is there
- none of the two parents are there


We don't distinguish between “in the nest” and “at the edge of the nest”. As
long as we can clearly make them out, we say “they're there”.


## Roadmap


1. Sample images from video recordings.
   - need enough samples for each class
2. Annotate samples.
3. Train MobileNet.
4. See how well MobileNet does.
5. Try deploying MobileNet to David's Raspberry Pi 3.


### TODO Mobilenet


1. Data Preprocessing
   - possibly need to resize the images to save computation time
   - Train-Test split
     - random sample form train set?
     - create test set from another day?
   - load images from folders and map class from folder-name
   - standardize data
2. Define Model:
   - use mobilenet
   - define head
3. Train model
   - finetune?
4. export trained
   - only weights?
   - full model, somehow?
