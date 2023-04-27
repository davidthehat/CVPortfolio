import numpy as np
import cv2 as cv
from picamera import PiCamera
from picamera.array import PiRGBArray
from matplotlib import pyplot as plt


resolution = (640, 480)

K=np.array([[459.45787954,0,340.78603026],[0,476.13449917, 246.87946725],[0,0,1]])

def capture_new_images():
    #takes as many images as requested by the user
    #returns a list of images0
    print("hit q to take the next image. press any other button to finish")
    imgs = []
    with PiCamera() as camera:
        camera.resolution = resolution
        camera.framerate = 30
        with PiRGBArray(camera, size=resolution) as rawCapture:
#             time.sleep(.1)
            camera.start_preview(fullscreen=False, window =(100, 20, 640, 480))
            cv.imshow('preview', np.zeros(resolution))
            cv.moveWindow('preview', 850, 100)
            while (cv.waitKey(0) & 0xFF == ord('q')):
                camera.capture(rawCapture, format='bgr')
                imgs.append(rawCapture.array)
                rawCapture.truncate(0)
                cv.imshow('preview', imgs[-1])
            cv.destroyWindow('preview')
            cv.waitKey(1)
            camera.stop_preview()
    return imgs

criteria = (cv.TERM_CRITERIA_EPS +cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

d1 = int(input('dim1:'))
d2 = int(input('dim2:'))

if (d1<=0 or d2<=0):
    print('using default')
else:

    objp=np.zeros((d1*d2, 3), np.float32)
    objp[:,:2]= np.mgrid[0:d1,0:d2].T.reshape(-1,2)

    objpoints =[]
    imgpoints =[]
    imgs=capture_new_images()
    for img in imgs:
        gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (d1, d2), None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            
            cv.drawChessboardCorners(img, (d1, d2), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey()
    cv.destroyAllWindows()
    print('calibrating camera')
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    #intrinsic is the intrisic matrix of the camera as detected by cv2
    K = mtx

print(K)

#now, take two more images and find the fundamental matrix

imgs = capture_new_images()
img1 = cv.cvtColor( imgs[0], cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor( imgs[1], cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

pts1=[]
pts2=[]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
        
        
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

print('found F')
print(F)

E = cv.transpose(K)@F@K
print('calculated E as transpose(K)@F@K')
print(E)

def drawlines(img1, img2, lines, pts1, pts2):
    """img1 - image on witch we draw the epilines for the points in img2
       lines - corresponding epilines"""
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    
    for r,pt1,pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[0]/r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0]*c/r[1])])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2 ,F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.show()

