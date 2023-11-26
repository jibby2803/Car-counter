# Car-counter

## Introduction:
Here is a source code for car counting using YOLOv8n model. It is a basic example of computer vision and object detection task.
In this code, I used sort.py file in an awesome github repository: https://github.com/abewley/sort/ for tracking task.
<p align="center">
  <img src="./result/result.gif" width=600><br/>
  <i>Result</i>
</p>

## How to use:
Run the following command
```bash
pip install -r requirements.txt
```
```bash
python detector.py
```
You can use my code for your own videos: In **detector.py**
* *Change the path of video to your own directory*
* *Change the path of mask image to your own directory (you can create mask images in Canva, make sure that your detected region should be white and other region should be black)*
* *Adjust the [x1, y1, x2, y2] according to your video and mask where (x1, y1), (x2, y2) are the coordinates of starting point and ending point of counting line*
* *Run detector.py*
