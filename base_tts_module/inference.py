import os
import time
from TTS.api import TTS
import torch


def generate_speech(text, model_path, config_path, output_folder="output_wavs"):
    os.makedirs(output_folder, exist_ok=True)

    timestamp = int(time.time() * 1000)
    output_path = os.path.join(output_folder, f"speech_{timestamp}.wav")
    torch.set_num_threads(4)  # Adjust based on your CPU cores (don't use all)
    torch.set_grad_enabled(False)

    tts = TTS(
        model_path=model_path, config_path=config_path, progress_bar=False, gpu=False
    )

    print(dir(tts))
    print("Model loading completed.")

    print("Starting preprocessing...")
    print(f"Text: {text}")
    print(f"Config path: {config_path}")

    tts.tts_to_file(text=text, file_path=output_path)

    print("Preprocessing completed.")

    return output_path


# Example usage
model_path = "configs/best_model.pth"
config_path = "configs/config.json"
output_folder = "output_folder"

text = "त्यसै गरी विश्वको कुनाकुनामा शान्तिको संदेश छरी अमर बन्न पुगेका शान्तिका अग्रदुत गौतम बुद्ध पनि मेरै देशका बासिन्दा हुन्।"

output_file = generate_speech(text, model_path, config_path, output_folder)
print(f"Generated speech saved at: {output_file}")
