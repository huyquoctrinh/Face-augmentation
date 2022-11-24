import mediapipe as mp
import cv2 
from get_landmark import getRightEyeRect, getLeftEyeRect, getLandmarks
from augment_utils import mixup_eyes
img = cv2.imread("test.png")

landmark = getLandmarks(img)
# print(landmark)

xRightEye, yRightEye, rightEyeWidth, rightEyeHeight, crop_eyeRight = getRightEyeRect(img,landmark)
xLeftEye, yLeftEye, leftEyeWidth, leftEyeHeight ,crop_eyeLeft = getLeftEyeRect(img, landmark)

print(img.shape)
print(xLeftEye,yRightEye)
print(crop_eyeLeft.shape)
# cv2.rectangle(img, (xRightEye, yRightEye),
#               (xRightEye + rightEyeWidth, yRightEye + rightEyeHeight), (200, 21, 36), 2)

# cv2.rectangle(img, (xLeftEye, yLeftEye),
#               (xLeftEye + leftEyeWidth, yLeftEye + leftEyeHeight), (200, 21, 36), 2)

res = mixup_eyes(img, crop_eyeLeft,xLeftEye, yLeftEye)

res = mixup_eyes(res, crop_eyeRight,xRightEye, yRightEye)
cv2.imwrite("save1.jpg",res)