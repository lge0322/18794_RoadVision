import pytesseract
from PIL import Image
import cv2
import numpy as np

img_path = "raw_imgs/stop_crop_2.png"

img_color = cv2.imread(img_path)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

_, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
cv2.imwrite("processed_imgs/test_img.png", binary_image)

results = pytesseract.image_to_string(binary_image, config='--psm 6')

print("Text detected:")
print(results)

lines = results.splitlines()
print(lines)
header = lines[0] 
data = lines[1:]  

detected_text = []
for line in data:
    cols = line.split('\t')
    if len(cols) == 12: 
        text = cols[-1] 
        conf = cols[-2] 
        if text.strip(): 
            detected_text.append(text)

h, w, _ = img_color.shape  

processed_image = img_color.copy()
boxes = pytesseract.image_to_boxes(binary_image,  config='--psm 6')

for b in boxes.splitlines():
    b = b.split(" ")
    processed_image = cv2.rectangle(processed_image, 
                            (int(b[1]), h-int(b[2])), 
                            (int(b[3]), h-int(b[4])), 
                            (0, 255, 0), 
                            2)
cv2.imwrite("processed_imgs/output_with_letter_boxes.png", processed_image)


word_image = img_color.copy()
data = pytesseract.image_to_data(word_image, config='--psm 6', output_type=pytesseract.Output.DICT)
n_boxes = len(data['text'])
for i in range(n_boxes):
    word = data['text'][i].strip()
    print(f"Word detected is: {word}")
    # Only process non-empty words and if the confidence is more than 60
    if word and (int(float(data['conf'][i])) > 60):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(word_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(word_image, word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imwrite("processed_imgs/output_with_word_boxes.png", word_image)