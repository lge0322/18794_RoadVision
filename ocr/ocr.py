import pytesseract
from PIL import Image
import cv2
import numpy as np

# Referenced from: https://github.com/h/pytesseract

# img_path = "sign_dataset/train/4.jpg"
img_path = "stop.png"
# img_path = "road_sign_4.png"
img_color = cv2.imread(img_path)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, None, fx=2, fy=2)


# Otsu Tresholding to find best threshold value
_, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)


# invert the image if the text is white and background is black
count_white = np.sum(binary_image > 0)
count_black = np.sum(binary_image == 0)
if count_black > count_white:
    binary_image = 255 - binary_image

binary_image = cv2.GaussianBlur(binary_image, (3, 3), 0)
kernel = np.ones((3, 3), np.uint8)
binary_image = cv2.erode(binary_image, kernel, iterations=1)
binary_image = cv2.dilate(binary_image, kernel, iterations=1)

# Crop the word "STOP" from the stop sign
edges = cv2.Canny(binary_image, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
height, width = binary_image.shape[:2]
center = (width // 2, height // 2)


# # take the first contour 
# cnt = contours[0] 
# rect = cv2.minAreaRect(cnt) 
# box = cv2.boxPoints(rect) 
# box = np.int0(box) 
# x1, y1 = box[0]
# x2, y2 = box[2]
# cv2.rectangle(binary_image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
# cv2.imwrite("BoundingRectangle.png", binary_image) 

image_with_boxes = binary_image.copy()
# Loop through each contour and draw a bounding box
for contour in contours:
    # rect = cv2.minAreaRect(contour) 
    # box = cv2.boxPoints(rect) 
    # box = np.int0(box) 
    # x1, y1 = box[0]
    # x2, y2 = box[2]
    # cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2) 

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color, thickness = 2

# Display the image with bounding boxes
cv2.imwrite("bbox.png", image_with_boxes)

# Filter contours by height similarity
height_tolerance = 10  # Adjust this tolerance for what counts as "similar" height
similar_height_contours = []

# Get bounding boxes for all contours and filter based on height
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if h >= 20:  # Filter out very small contours if necessary
        similar_height_contours.append((x, y, w, h))

# Sort contours by their vertical position (y-coordinate)
similar_height_contours.sort(key=lambda c: c[1])

# Group contours into rows based on similar y-coordinate ranges
rows = []
row = []
prev_y = -1
for (x, y, w, h) in similar_height_contours:
    if prev_y == -1 or abs(y - prev_y) < height_tolerance:  # Same row
        row.append((x, y, w, h))
    else:  # New row
        rows.append(row)
        row = [(x, y, w, h)]  # Start new row
    prev_y = y

# Add the last row
if row:
    rows.append(row)

# Find a row with 4 contours
for row in rows:
    if len(row) >= 4:  # Look for rows with at least 4 contours
        # Sort the row by x-coordinate to get the left-to-right order
        row.sort(key=lambda c: c[0])
        
        # Get the bounding box for the group of 4 contours
        x_min = min([c[0] for c in row[:4]])
        y_min = min([c[1] for c in row[:4]])
        x_max = max([c[0] + c[2] for c in row[:4]])
        y_max = max([c[1] + c[3] for c in row[:4]])

        # Crop the image based on the bounding box
        cropped_image = binary_image[y_min:y_max, x_min:x_max]

        # Display the cropped image
        cv2.imwrite('cropped.png', cropped_image)

cv2.imwrite("bin_img_stop.png", binary_image)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(cropped_image, kernel, iterations=1)
cv2.imwrite('eroded.png', eroded)


thinned_image = cv2.ximgproc.thinning(255-cropped_image)
cv2.imwrite('thinned.png', 255-thinned_image)


print("Text detected:")
print(pytesseract.image_to_string(eroded))

h, w, c = img_color.shape
boxes = pytesseract.image_to_boxes(img_color) 
for b in boxes.splitlines():
    b = b.split(' ')
    img_color = cv2.rectangle(img_color, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imwrite('output.png', img_color)
