# Face Expression Recognition
&nbsp;&nbsp; Facial expression recognition using CNN model trained on fer2013 dataset.

### Requirements:
&nbsp;&nbsp; opencv-python==4.2.0 </br>
&nbsp;&nbsp; matplotlib==3.3.0 </br>
&nbsp;&nbsp; pandas==1.0.5 </br>
&nbsp;&nbsp; numpy==1.19.1 </br>
&nbsp;&nbsp; keras==2.3.1 </br>

### Demo:
&nbsp;&nbsp; Demo ipython notebook - [demo.ipynb](https://github.com/keshavoct98/Face-expression-recognition/blob/master/demo.ipynb) </br>
<pre><code> #For images 
 python demo.py --input [path_to_input_image]
 
 #Example
 python demo.py --input data/inputs/0.jpg </pre></code>
  
<pre><code> #For videos
 python demo.py --input [path_to_input_video] --video 
 
 #Example
 python demo.py --input data/inputs/vid0.mp4 </pre></code>
 
### Result:
![data/outputs/output.gif](https://github.com/keshavoct98/Face-expression-recognition/blob/master/data/outputs/output.gif)
![data/outputs/out1.jpg](https://github.com/keshavoct98/Face-expression-recognition/blob/master/data/outputs/out1.jpg)
![data/outputs/out2.jpg](https://github.com/keshavoct98/Face-expression-recognition/blob/master/data/outputs/out2.jpg)

### How to train:
1. Download dataset from this link - https://www.kaggle.com/deadskull7/fer2013
2. Extract the content of zip inside 'data' folder.
3. Run below command -
<pre><code> python train.py --path [path_to_csv_file] --epochs [no_of_epochs]
 #Example - python train.py --path data/fer2013.csv --epochs 15 </pre></code>
 
### References
1. https://www.kaggle.com/deadskull7/fer2013
2. https://www.kaggle.com/deadskull7/facreco-90-14-10-epochs
3. https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
