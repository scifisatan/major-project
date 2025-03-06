from voice_conversion_module import convert_voice


def test_voice_conversion():
    import torch

    source_audio_path = "tests/mcd_test.wav"
    reference_speaker_path = "tests/mcd_test_ref.wav"
    output_path = "output/converted.wav"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        converted_audio_path = convert_voice(
            source_audio_path=source_audio_path,
            reference_speaker_path=reference_speaker_path,
            output_path=output_path,
            device=device,
        )
        print(f"Voice conversion complete! Output saved to: {converted_audio_path}")
    except Exception as e:
        print(f"An error occurred during voice conversion: {e}")


if __name__ == "__main__":
    test_voice_conversion()
