import cv2
import mediapipe as mp
import math
from flask import Flask, render_template, Response, request, redirect, url_for
import json
import pyttsx3
from threading import Thread, Lock
import time
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# Initialize Flask app
app = Flask(__name__)
# Initialize mediapipe pose class
mp_pose = mp.solutions.pose
# Setting up the Pose function
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

# 添加一个全局锁和最后播放时间记录
speak_lock = Lock()
last_speak_time = 0

# 在全局范围内加载 json 数据
with open("sum_data.json", 'r', encoding='utf-8') as f:
    sum_data = json.load(f)
angle_cache = {}

# Add these configurations after app initialization
UPLOAD_FOLDER = 'static/detect'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def speak_text(text):
    """在后台线程播放文字转语音"""
    global last_speak_time
    
    # 检查是否距离上次播放已经过了至少2秒
    current_time = time.time()
    with speak_lock:
        if current_time - last_speak_time < 2:  # 2秒冷却时间
            return
        last_speak_time = current_time
    
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"Error speaking text: {e}")

def detectPose(image, pose):
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                (landmark.z * width)))
    
    
    return output_image,landmarks


def calculateAngle(landmark1, landmark2, landmark3):
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    #radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
    #angle = np.abs(radians*180.0/np.pi)
    

    
    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

def AngleProcess(label_name, List_input):
    # 使用缓存的角度数据
    if label_name not in angle_cache:
        angle_cache[label_name] = json.load(open(f"{label_name}.json"))
    angle = angle_cache[label_name]
    
    # 创建翻转后的角度列表
    # 左右对称的部位角度互换，并对某些角度进行360度补偿
    angle_2 = [angle[1], angle[0], angle[3], angle[2], 360-angle[5], 360-angle[4]]
    
    # 定义允许的最大误差
    max_error = 50
    
    # 检查原始角度
    errors = [abs(i - j) for i, j in zip(angle, List_input)]
    if all(error <= max_error for error in errors):
        return True
        
    # 检查翻转后的角度
    errors_2 = [abs(a - b) for a, b in zip(angle_2, List_input)]
    if all(error <= max_error for error in errors_2):
        return True
        
    return False


def classifyPose(landmarks, output_image):
    with open("sum_data.json", 'r', encoding='utf-8') as f:
        sum_data = json.load(f)
    label = 'Unknown Pose'
    color = (0, 0, 255)
        
    # 计算左臂的角度 (左肩-左肘-左手腕)
    left_arm_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # 计算右臂的角度 (右肩-右肘-右手腕)
    right_arm_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # 计算左上身的角度 (左肘-左肩-左臀)
    left_body_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # 计算右上身的角度 (右臀-右肩-右肘)
    right_body_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # 计算左腿的角度 (左臀-左膝-左踝)
    left_leg_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # 计算右腿的角度 (右臀-右膝-右踝)
    right_leg_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # 将所有角度放入列表中用于后续处理
    List_input = [left_arm_angle, right_arm_angle, left_body_angle, right_body_angle, left_leg_angle, right_leg_angle]
    
    for label_name in sum_data:
        if AngleProcess(label_name,List_input)==True:      
            label = label_name
            color = (0, 255, 0)
            
            # 从本地文件读取并播放说明
            try:
                with open('pose_instructions.json', 'r', encoding='utf-8') as f:
                    instructions = json.load(f)
                    instruction = instructions.get(label, f"当前姿势是 {label}")
                    Thread(target=speak_text, args=(instruction,), daemon=True).start()
            except Exception as e:
                print(f"Error reading instructions: {e}")
                Thread(target=speak_text, args=(f"当前姿势是 {label}",), daemon=True).start()
    
    # 使用 cv2.putText 显示中文
    img_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 加载中文字体（需要指定字体文件路径）
    fontpath = "simhei.ttf"  # 请确保此字体文件存在
    font = ImageFont.truetype(fontpath, 32)
    
    # 在图片上绘制中文文本
    draw.text((10, 30), label, font=font, fill=color[::-1])  # OpenCV的BGR转换为RGB
    
    # 转换回OpenCV格式
    output_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return output_image, label

def webcam_feed():
    # Initialize the VideoCapture object to read from the webcam
    #camera_video = cv2.VideoCapture(0)
    camera_video = cv2.VideoCapture("show_video.mp4")
    camera_video.set(3, 1380)
    camera_video.set(4, 960)

    try:
        while camera_video.isOpened():
            # Read a frame
            ok, frame = camera_video.read()

            if not ok:
                continue

            # Flip the frame horizontally for natural (selfie-view) visualization
           # frame = cv2.flip(frame, 1)

            # Get the width and height of the frame
            frame_height, frame_width, _ = frame.shape

            # Resize the frame while keeping the aspect ratio
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            # Perform Pose landmark detection
            frame, landmarks = detectPose(frame, pose)

            if landmarks:
                # Perform the Pose Classification
                frame, _ = classifyPose(landmarks, frame)

            # Convert the frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera_video.release()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pose_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
        
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    target_height = 640
    scale = target_height / height
    target_width = int(width * scale)
    image = cv2.resize(image, (target_width, target_height))
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print("Error: No pose landmarks detected. Please ensure the image contains a clearly visible person.")
        return None
    
    # Convert landmarks to our format
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.append((
            int(landmark.x * target_width),
            int(landmark.y * target_height),
            landmark.z * target_width
        ))
    
    try:
        angles = []
        # Left arm angle
        angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]))
        # Right arm angle
        angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]))
        # Left body angle
        angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]))
        # Right body angle
        angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]))
        # Left leg angle
        angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]))
        # Right leg angle
        angles.append(calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
        
        # Save debug image with landmarks
        debug_image = image.copy()
        mp_drawing.draw_landmarks(
            debug_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        debug_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_landmarks.jpg')
        cv2.imwrite(debug_path, debug_image)
        
        return angles
        
    except Exception as e:
        print(f"Error calculating angles: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yoga_try')
def yoga_try():
    # Load all poses from sum_data.json
    with open('sum_data.json') as f:
        poses = json.load(f)
    return render_template('yoga_try.html', poses=poses)

@app.route('/video_feed1')
def video_feed1():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload')
def upload():
    with open('sum_data.json', 'r') as f:
        poses = json.load(f)
    return render_template('upload.html', poses=poses)

@app.route('/upload_pose', methods=['POST'])
def upload_pose():
    if 'pose_image' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['pose_image']
    pose_name = request.form['pose_name']
    
    # 检查姿势名称是否已存在
    with open('sum_data.json', 'r') as f:
        sum_data = json.load(f)
    if pose_name in sum_data:
        return '该姿势名称已存在', 400
    
    if file.filename == '':
        return 'No file selected', 400
    
    if file and allowed_file(file.filename):
        # 保存图片文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image and get angles
        angles = process_pose_image(file_path)
        if angles is None:
            os.remove(file_path)
            return 'No pose detected in image', 400
            
        # 检查是否存在相同角度的姿势
        for existing_pose in sum_data:
            with open(f'{existing_pose}.json', 'r') as f:
                existing_angles = json.load(f)
            if is_similar_pose(angles, existing_angles):
                os.remove(file_path)
                return f'检测到相似姿势: {existing_pose}', 400
        
        # 处理成功后，重命名文件
        new_filename = f"{pose_name}.jpg"
        new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        os.rename(file_path, new_file_path)
        
        # Save angles to JSON
        with open(f'{pose_name}.json', 'w') as f:
            json.dump(angles, f)
        
        # Update sum_data.json
        sum_data.append(pose_name)
        with open('sum_data.json', 'w') as f:
            json.dump(sum_data, f)
        
        return redirect(url_for('yoga_try'))
    
    return 'Invalid file type', 400

# 添加新的路由处理删除请求
@app.route('/delete_pose/<pose_name>')
def delete_pose(pose_name):
    try:
        # 删除图片文件
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{pose_name}.jpg")
        if os.path.exists(image_path):
            os.remove(image_path)
            
        # 删除JSON文件
        json_path = f"{pose_name}.json"
        if os.path.exists(json_path):
            os.remove(json_path)
            
        # 从sum_data.json中移除
        with open('sum_data.json', 'r') as f:
            sum_data = json.load(f)
        if pose_name in sum_data:
            sum_data.remove(pose_name)
        with open('sum_data.json', 'w') as f:
            json.dump(sum_data, f)
            
        return redirect(url_for('upload'))
    except Exception as e:
        return f'Error deleting pose: {str(e)}', 500

# 添加新的辅助函数来检查姿势相似度
def is_similar_pose(angles1, angles2, threshold=10):
    """
    比较两组角度是否相似
    threshold: 允许的角度差异阈值（度）
    """
    return all(abs(a - b) <= threshold for a, b in zip(angles1, angles2))

if __name__ == '__main__':
    app.run(debug=True)
