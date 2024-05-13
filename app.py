from flask import Flask, render_template, request, send_file
from circle_detection_module import find_hough_circles
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
@app.route('/detect',  methods=['GET', 'POST'])
def detect():
    if 'file' not in request.files:
        return render_template('detect.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('detect.html', error='No selected file')

    if file:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Perform circle detection
        input_img = cv2.imread(image_path)
        edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        edge_image = cv2.Canny(edge_image, 100, 200)
        circle_img, circles = find_hough_circles(input_img, edge_image, r_min=10, r_max=200, delta_r=1, num_thetas=100, bin_threshold=0.4)

        # Save the result image
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image.png')
        cv2.imwrite(result_image_path, circle_img)

        # Render the result page
        return render_template('detect.html', result_image=result_image_path)

    return render_template('detect.html', error=None)

@app.route('/')
def home():
    return render_template('home.html', error=None)

@app.route('/detect_circles')
def detect_circles():
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image.png')
    return send_file(result_image_path, mimetype='image/png')
@app.route('/about')
def about():
    return render_template('about.html', error=None)
@app.route('/me')
def me():
    return render_template('me.html', error=None)
if __name__ == '__main__':
    app.run(debug=True)
