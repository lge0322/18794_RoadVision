Modifications from the original package
- helpers/labels.py: changed the labels to binary classifications
- preparations/json2labelImg.py: changed the key "unlabeled" to "background"

Additions
- modify_json.py: based on labels.py, all the labels in the json files are changed 
into either "background" or "traffic sign."
- create_img.py: check if the masks are correct via creating a colored image

Steps
1. Do "export CITYSCAPES_DATASET=/path/to/cityscapes/dataset" so that you can run
the script from the root folder
2. Run modify_json to modify all the JSON files in the dataset
3. Run create_img to see masks in individual images


