import os
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from glob import glob
import jsonlines
import shutil
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms as T
from transformers import BlipProcessor, BlipForConditionalGeneration

register_heif_opener()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_size = 768
images_path = 'photos/'
blip_labels_path = 'blip-labels-test/train'
image_ext = ['jpg', 'png', 'jpeg', 'heic']

start_string = 'a cat'
replace_string = 'Masha'

im_transforms = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size)
])

images = []
for ext in image_ext:
    images.extend(glob(str(Path(images_path, f'*.{ext}'))))

os.makedirs(blip_labels_path, exist_ok=True)

with jsonlines.open(Path(blip_labels_path, 'metadata.jsonl'), mode='w') as writer:
    for image in tqdm(images):
        raw_image = Image.open(image).convert('RGB')
        raw_image = ImageOps.exif_transpose(raw_image)
        croped_image = im_transforms(raw_image)
        croped_image.save(str(Path(blip_labels_path, Path(image).name)))

        inputs = processor(croped_image, start_string, return_tensors="pt")
        out = model.generate(**inputs)
        label_str = processor.decode(out[0], skip_special_tokens=True)
        label_str = label_str.replace(start_string, replace_string)
        
        writer.write({'file_name': Path(image).name, 'text': label_str})