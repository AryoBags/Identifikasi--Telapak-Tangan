from flask import Flask, render_template, request
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import pickle
import time

app = Flask(__name__)

# Load the model and the scaler
model = pickle.load(open("model-tanganektra.pkl", "rb"))
scaler = pickle.load(open("scaler-tanganektra.pkl", "rb"))  # Ensure the scaler is saved and loaded

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediction', methods=["POST"])
def prediction():
    img = request.files['img']
    img_path = "static/img.jpg"
    img.save(img_path)

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    
    img_resized = cv2.resize(image, dsize=(480, 640), interpolation=cv2.INTER_CUBIC)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    input_image = Image.fromarray(img_resized)
    output_image = remove(input_image)
    
    if output_image.mode == 'RGBA':
        output_image = output_image.convert("RGB")
    img_original = np.array(output_image)
    if len(img_original.shape) == 3:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    
    h, w = img_original.shape
    img = np.zeros((h + 160, w), np.uint8)
    img[80:-80, :] = img_original
    
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    M = cv2.moments(th)
    h, w = img.shape
    x_c = int(M['m10'] / M['m00'])
    y_c = int(M['m01'] / M['m00'])
    
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)
    erosion = cv2.erode(th, kernel, iterations=1)
    boundary = th - erosion
    cnt, _ = cv2.findContours(boundary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = cnt[0].reshape(-1, 2)
    
    left_id = np.argmin(cnt.sum(-1))
    cnt = np.concatenate([cnt[left_id:, :], cnt[:left_id, :]])
    
    dist_c = np.sqrt(np.square(cnt - [x_c, y_c]).sum(-1))
    f = np.fft.rfft(dist_c)
    cutoff = 15
    f_new = np.concatenate([f[:cutoff], 0 * f[cutoff:]])
    dist_c_1 = np.fft.irfft(f_new)
    
    derivative = np.diff(dist_c_1)
    sign_change = np.diff(np.sign(derivative)) / 2
    minimas = cnt[np.where(sign_change > 0)[0]]
    v1, v2 = minimas[-1], minimas[-3]
    
    theta = np.arctan2((v2 - v1)[1], (v2 - v1)[0]) * 180 / np.pi
    center = (int((v1[0] + v2[0]) // 2), int((v1[1] + v2[1]) // 2))
    R = cv2.getRotationMatrix2D(tuple(center), theta, 1)
    img_r = cv2.warpAffine(img, R, (w, h))
    v1 = (R[:,:2] @ v1 + R[:,-1]).astype(int)
    v2 = (R[:,:2] @ v2 + R[:,-1]).astype(int)
    
    ux = v1[0]
    uy = v1[1] + (v2 - v1)[0] // 3
    lx = v2[0]
    ly = v2[1] + 4 * (v2 - v1)[0] // 3
    roi = img_r[uy:ly, ux:lx]
    
    roi_for_hu_moments = roi.copy()
    hu_moments = cv2.HuMoments(cv2.moments(roi_for_hu_moments)).flatten()
    hu_moments = [moment for moment in hu_moments]
    
    canny_edges = cv2.Canny(roi, 60, 60)
    lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold = 20, minLineLength = 5, maxLineGap = 10)
    
    number_of_lines = 0
    num_canny_edges = np.sum(canny_edges > 0)
    if lines is not None:
        number_of_lines = len(lines)
        
    cv2.imwrite('static/roi.jpg', roi)
    cv2.imwrite('static/canny_edges.jpg', canny_edges)
    hough_lines_image = np.zeros_like(roi)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_lines_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv2.imwrite('static/hough_lines.jpg', hough_lines_image)
    
    # Predict using the model
    features = np.array([number_of_lines, num_canny_edges] + hu_moments).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    predicted_label = model.predict(features_scaled)[0]

    timestamp = int(time.time())
    
    return render_template("prediction.html", data=number_of_lines, num_canny_edges=num_canny_edges,
                           hu1=hu_moments[0], hu2=hu_moments[1], hu3=hu_moments[2], hu4=hu_moments[3], 
                           hu5=hu_moments[4], hu6=hu_moments[5], hu7=hu_moments[6], label=predicted_label,
                           img_path=f'{img_path}?{timestamp}', 
                           roi_path=f'static/roi.jpg?{timestamp}', 
                           canny_path=f'static/canny_edges.jpg?{timestamp}', 
                           hough_path=f'static/hough_lines.jpg?{timestamp}')

if __name__ == '__main__':
    app.run(debug=True)
