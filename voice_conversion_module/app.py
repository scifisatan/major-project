import os
from voice_conversion_module.se_extractor import get_se
from voice_conversion_module.converter import ToneColorConverter


def convert_voice(
    source_audio_path,
    reference_speaker_path,
    output_path,
    device="cuda:0",
    checkpoints_dir=os.path.join(os.path.dirname(__file__), "checkpoints/converter"),
    tau=0.3,
    processed_dir=os.path.join(os.path.dirname(__file__), "processed"),
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tone_color_converter = ToneColorConverter(
        f"{checkpoints_dir}/config.json", device=device
    )
    tone_color_converter.load_ckpt(f"{checkpoints_dir}/checkpoint.pth")

    source_speaker_embeddings, _ = get_se(
        source_audio_path, tone_color_converter, target_dir=processed_dir, vad=True
    )

    target_speaker_embeddings, _ = get_se(
        reference_speaker_path, tone_color_converter, target_dir=processed_dir, vad=True
    )

    tone_color_converter.convert(
        audio_src_path=source_audio_path,
        src_se=source_speaker_embeddings,
        tgt_se=target_speaker_embeddings,
        output_path=output_path,
        tau=tau,
    )

    return output_path
