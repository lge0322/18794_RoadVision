import json, os, glob
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.preparation.json2labelImg import createLabelImage

# Reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2labelImg.py
inJson = "/Users/gaeunlee/Desktop/F2024/18794/Project/Cityscape_data/dataset/gtFine/train/aachen/aachen_000000_000019_gtFine_polygons.json"
annotation = Annotation()
annotation.fromJsonFile(inJson)
labelImg = createLabelImage(annotation, "color")
labelImg.save("/Users/gaeunlee/Desktop/F2024/18794/Project/Cityscape_data/test.png")

# Reference from createTrainIdLabelImgs.py
def main():
    # Where to look for Cityscapes
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

    searchFine   = os.path.join( cityscapesPath , "gtFine"   , "*" , "*" , "*_gt*_polygons.json" )
    searchCoarse = os.path.join( cityscapesPath , "gtCoarse" , "*" , "*" , "*_gt*_polygons.json" )

    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()
    filesCoarse = glob.glob( searchCoarse )
    filesCoarse.sort()

    files = filesFine + filesCoarse

    if not files:
        printError( "Did not find any files. Please consult the README." )

    print("Processing {} annotation files".format(len(files)))

    traffic_sign_label = "traffic sign"  # Add more labels if necessary
    progress = 0
    print("Progress: {:>3} %".format(progress * 100 / len(files)), end=' ')
    
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)

        objects_arr = []

        for obj in data["objects"]:
            
            if obj["label"] == traffic_sign_label:
                # keep the traffic sign labels
                objects_arr.append(obj)
            else:
                # Change non-traffic sign labels to "background"
                objects_arr.append({
                    "label": "background",
                    "polygon": obj["polygon"]
                })

        data["objects"] = objects_arr

        with open(f, 'w') as file:
            json.dump(data, file, indent=4)

        progress += 1
        print("\rProgress: {:>3} %".format(progress * 100 / len(files)), end=' ')
    
    print("\nAll JSON files processed and updated.")

if __name__ == "__main__":
    main()
