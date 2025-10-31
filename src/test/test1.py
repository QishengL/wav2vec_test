import datasets
import re
from datasets import load_dataset
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS
from transformers import AutoProcessor, AutoModelForPreTraining


processor = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-xls-r-300m")


'''
raw_datasets = datasets.DatasetDict()

raw_datasets["train"] = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "tr",
    split="train+validation",
    trust_remote_code=True,
)

raw_datasets["eval"] = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "tr",
    split="test",
    trust_remote_code=True,
)

raw_datasets["train"] = raw_datasets["train"].select(range(2))

raw_datasets["eval"] = raw_datasets["eval"].select(range(2))
    
import re
def preprocess_datasets(raw_datasets):
    # 示例：清理字符
    backend = BACKENDS["espeak"]("tr", language_switch="remove-flags")
    chars_to_ignore = ["…","’",",", "?", ".", "!", "-", ";", ":", "\"", "“", "%", "‘", "”","�", "(", ")", "'"]
    if chars_to_ignore:
        if isinstance(chars_to_ignore, str):
            chars_to_ignore = chars_to_ignore.split()
        chars_to_ignore_regex = f"[{''.join(re.escape(c) for c in chars_to_ignore)}]"
    else:
        chars_to_ignore_regex = None

    def remove_special_characters(batch):
        text = batch['sentence']
        if chars_to_ignore_regex is not None:
            text = re.sub(chars_to_ignore_regex, "", text)
        batch["target_text"] = text.lower() + " "
        return batch
    
    def phonemize_data(batch):
        text = batch['sentence']
        separator = Separator(phone=' ', word="", syllable="")
        phonemes = backend.phonemize(
            [text],
            separator=separator,
        )
        processed_text = phonemes[0].strip()
        print(processed_text)
        batch["target_text"] = processed_text
        return batch
    return raw_datasets.map(phonemize_data, desc="Clean text")

raw_datasets = preprocess_datasets(raw_datasets)







backend = BACKENDS["espeak"]("tr", language_switch="remove-flags")
separator = Separator(phone=' ', word="", syllable="")
phonemes = backend.phonemize(
    ["Sorun yok, tamam mı?  "],
    separator=separator,
)
processed_text = phonemes[0].strip()
'''