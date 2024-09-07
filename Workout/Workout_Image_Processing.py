import pytesseract
from PIL import Image, ImageEnhance
import pandas as pd
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

#Preprocessing added to enhance and convert to grayscale for OCR
def preprocess_image(image_path):
    with Image.open(image_path) as img:
        # Convert to grayscale
        gray = img.convert('L')
        # Enhance the image
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2)
        # Apply thresholding
        thresholded = enhanced.point(lambda p: p > 128 and 255)
        return thresholded

def extract_text_from_image(image_path):
    # Open an image file
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img)
    return text

def parse_text(text):
    # Define regex patterns to extract relevant information
    date_pattern = r'\w+ \d\d?,\s?202\d'
    exercise_pattern = r'(%|x|iv)\s+(.*?)\s+(Instructions)'
    description_pattern = r'Instructions\s+((.|\s)*?)\s+(Results|Add|Update)'
    weight_pattern = r'(\s(Add results)|Update results|Results\s+((.|\s)*?)\s+\w?(Update|Has attac)|Results\s+((.|\s)*?)$)'

    date = re.search(date_pattern, text)
    exercise = re.search(exercise_pattern, text)
    description = re.search(description_pattern, text)
    weight = re.search(weight_pattern, text)

    return {
        'Date': date.group(0) if date else None,
        'Exercise': exercise.group(2) if exercise else None,
        'Description': description.group(1) if description else None,
        'Result': weight.group(2) if weight.group(2) else weight.group(3) if weight.group(3) else weight.group(6)
    }

# Directory containing the images
image_folder = 'Exercises/Workout/images'
images = sorted([image_folder + "/" + img for img in os.listdir(image_folder) if img.endswith(".PNG")])

data = []
# for image_path in images:
def process_image(image_path):
    print(image_path)
    text = extract_text_from_image(image_path)
    parsed_data = parse_text(text)
    data.append(parsed_data)

# Parallel processing with ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    future_to_image = {executor.submit(process_image, image_path): image_path for image_path in images}
    for future in as_completed(future_to_image):
        result = future.result()
        if result:
            data.append(result)

# Create a DataFrame
df = pd.DataFrame(data)

# Save to a CSV file
df.to_csv('Exercises/Workout/workout_data.csv', index=False)