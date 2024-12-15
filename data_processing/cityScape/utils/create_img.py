from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.csHelpers import printError
from cityscapesscripts.preparation.json2labelImg import createLabelImage

# testing
inJson = "/Users/gaeunlee/Desktop/F2024/18794/Project/Cityscape_data/dataset/gtFine/train/aachen/aachen_000000_000019_gtFine_polygons.json"
annotation = Annotation()
annotation.fromJsonFile(inJson)
labelImg = createLabelImage(annotation, "color")
labelImg.save("/Users/gaeunlee/Desktop/F2024/18794/Project/Cityscape_data/test.png")
