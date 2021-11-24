from dalle_mini.model import CustomFlaxBartForConditionalGeneration
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('flax-community/dalle-mini')
model = CustomFlaxBartForConditionalGeneration.from_pretrained('flax-community/dalle-mini')
