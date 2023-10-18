import csv
import os
import glob
from PIL import Image
import math
from datetime import datetime

def calculate_age(birth_date, photo_year):
    year, month, day = int(birth_date.split('-')[0]), int(birth_date.split('-')[1]), int(birth_date.split('-')[2])
    if month == 0 or day == 0 or year == 0:
        print(f"Ignored invalid birth date: {birth_date}")
        return None
    birth_datetime = datetime.strptime(birth_date, '%Y-%m-%d')
    age_at_photo = photo_year - birth_datetime.year
    return age_at_photo

def process_image(image_path):
    # Open image
    img = Image.open(image_path)

    # Check if image is square
    if img.size[0] == img.size[1]:
        # Resize to 256x256 and overwrite the image
        # img_resized = img.resize((256, 256))
        # img_resized.save(image_path)
        return image_path
    else:
        print(f"Image {image_path} is not square.")
        return None

if __name__ == '__main__':
    # Assuming that parent_dir is the directory containing all your image folders
    parent_dir = 'Z:\data\Face\imdb_crop'

    # Get the list of all directories and sort
    folders = sorted([folder.path for folder in os.scandir(parent_dir) if folder.is_dir()])

    # Iterate over all directories
    for folder in folders:
        print(folder)
        # Prepare data for CSV
        data = []
        # Get list of all images and sort
        image_paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))

        # Process images
        for image_path in image_paths:
            sort_image_path = process_image(image_path)

            if sort_image_path is not None:
                # Extract filename without extension
                filename = os.path.splitext(os.path.basename(sort_image_path))[0]
                actual_folder = os.path.splitext(sort_image_path)[0].split('\\')[-2]
                # Extract id, birth date, and photo year from filename
                person_id = filename.split('_')[0]
                birth_date = filename.split('_')[2]
                photo_year = int(filename.split('_')[3])

                # Calculate age
                age = calculate_age(birth_date, photo_year)
                if age is not None:
                    # Calculate age group
                    age_group = math.floor(age/10)
                    if age_group >= 10:
                        print(filename)
                    # Append id, filename, and age to data
                    data.append((os.path.join(actual_folder, filename), age_group, age, person_id))

        # Write data to CSV
        with open('image_ages.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if actual_folder == '00':
                writer.writerow(['image_id', 'age_group', 'age', 'person_id'])

            writer.writerows(data)
