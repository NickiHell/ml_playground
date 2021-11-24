import os
import random
from uuid import uuid4

import jax
import numpy as np
from PIL import Image
from dalle_mini.model import CustomFlaxBartForConditionalGeneration
from torchvision.utils import save_image
from tqdm.notebook import tqdm
from transformers import BartTokenizer
from vqgan_jax.modeling_flax_vqgan import VQModel

os.makedirs('images', exist_ok=True)

# make sure we use compatible versions
DALLE_REPO = 'flax-community/dalle-mini'
VQGAN_REPO = 'flax-community/vqgan_f16_16384'

# set up tokenizer and model
tokenizer = BartTokenizer.from_pretrained(DALLE_REPO)
model = CustomFlaxBartForConditionalGeneration.from_pretrained(DALLE_REPO)

if __name__ == '__main__':
    prompt = 'picture of a waterfall under the sunset'

    tokenized_prompt = tokenizer(prompt, return_tensors='jax', padding='max_length', truncation=True, max_length=128)

    n_predictions = 8

    # create random keys
    seed = random.randint(0, 2 ** 32 - 1)
    key = jax.random.PRNGKey(seed)
    subkeys = jax.random.split(key, num=n_predictions)

    encoded_images = [model.generate(**tokenized_prompt, do_sample=True, num_beams=1, prng_key=subkey) for subkey in
                      tqdm(subkeys)]
    encoded_images[0]

    encoded_images = [img.sequences[..., 1:] for img in encoded_images]
    encoded_images[0]

    encoded_images[0].shape

    vqgan = VQModel.from_pretrained(VQGAN_REPO)
    decoded_images = [vqgan.decode_code(encoded_image) for encoded_image in tqdm(encoded_images)]
    decoded_images[0]

    clipped_images = [img.squeeze().clip(0., 1.) for img in decoded_images]

    images = [Image.fromarray(np.asarray(img * 255, dtype=np.uint8)) for img in clipped_images]

    [save_image(images[x], f'images/{uuid4()}') for x in range(images)]
