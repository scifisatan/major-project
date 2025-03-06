from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import os
import shutil
from voice_conversion_module import convert_voice

app = FastAPI()

app.add_middleware(
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Ensure the uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/generate_audio")
async def generate_audio(
    text: str = Form(...), reference_speaker: UploadFile = File(...)
):
    """
    Accepts a text and an audio file (reference_speaker), saves the audio,
    and returns the saved file as a response.
    """

    file_location = os.path.join(UPLOAD_DIR, reference_speaker.filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(reference_speaker.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save the file.")

    # Call your voice conversion script
    print("Calling voice conversion script...")
    reference_speaker_path = file_location
    output_path = voice_conversion_script.generate_audio(reference_speaker_path, text)

    # Return the saved file as a FileResponse.
    # Using the content type from the uploaded file makes the response flexible.
    return FileResponse(
        path=output_path,
        media_type=reference_speaker.content_type,
        filename="converted_audio.wav",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8888, reload=True)
