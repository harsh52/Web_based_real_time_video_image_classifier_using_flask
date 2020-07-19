# main.py# import the necessary packages
from flask import Flask, render_template, Response, request
from camera__final import VideoCamera
from image__final import ImageClassify
import os
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

'''
I am looking for a job.
kindly contact me if you want to help me.
Thanks..
'''

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')


def dir_last_updated(folder):

    return str(max(os.path.getmtime(os.path.join(root_path, f))for root_path, dirs, files in os.walk(folder)for f in files))


@app.route("/video_forward/")
def video_redirect():
	return render_template('video_classify.html')

@app.route("/image_forward/")
def image_redirect():
    #Moving forward code
   # forward_message = "Moving Forward..."
    return render_template('image_classify.html')


@app.route("/view_train_img/", methods=['POST'])
def trained_image():
	return render_template('view_img.html',last_updated=dir_last_updated('./static'));

@app.route("/upload", methods=['POST'])
def upload():
	target = os.path.join(APP_ROOT,'images/')
	print(target)
	if not os.path.isdir(target):
		os.mkdir(target)
	for file in request.files.getlist("file"):
		#print(file)
		filename = file.filename
		destination = "/".join([target,filename])
		#print(destination)
		
		#print(ret)
		file.save(destination)
		ret=ImageClassify().get_frame(destination)
		destination = ''
	return render_template("complete.html")


def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')        
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)
