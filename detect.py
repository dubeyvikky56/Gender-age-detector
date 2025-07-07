import os
from deepface import DeepFace

# Folder where image is located
folder_path = r"D:\gad"

# Valid image extensions
valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']

# Auto-select the first valid image
image_path = None
for file in os.listdir(folder_path):
    if os.path.splitext(file)[1] in valid_extensions:
        image_path = os.path.join(folder_path, file)
        break

# Raise error if no image found
if image_path is None:
    raise FileNotFoundError("No image file found in the folder.")

# Run DeepFace analysis
result = DeepFace.analyze(img_path=image_path, actions=['age', 'gender'])

# Output results
print("Image Analyzed:", image_path)
print("Predicted Age:", result[0]['age'])
print("Predicted Gender:", result[0]['gender'])
