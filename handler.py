import runpod
import base64
import io
from groq import Groq
from openai import OpenAI
import os
import asyncio

# Initialize AI clients
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

DEFAULT_MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are an AI assistant created by Sayed Raheel. Keep answers concise and helpful."

async def process_llm(text: str) -> str:
    """Process text through LLM"""
    try:
        chat_completion = await asyncio.to_thread(
            groq_client.chat.completions.create,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            model=DEFAULT_MODEL,
            temperature=0.5,
            max_tokens=2048,
            top_p=1,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"LLM processing error: {str(e)}")

async def generate_speech(text: str) -> str:
    """Generate text-to-speech"""
    try:
        # Generate TTS response
        tts_response = await asyncio.to_thread(
            openai_client.audio.speech.create,
            model="tts-1",
            voice="onyx",
            input=text
        )
        
        # Convert to base64
        audio_response = io.BytesIO()
        for chunk in tts_response.iter_bytes():
            audio_response.write(chunk)
        return base64.b64encode(audio_response.getvalue()).decode()
    except Exception as e:
        raise Exception(f"TTS generation error: {str(e)}")

async def process_audio_input(audio_base64: str):
    """Process audio input"""
    try:
        # Decode audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Using temp file for Groq processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            # Transcribe
            translation = await asyncio.to_thread(
                groq_client.audio.translations.create,
                file=(temp_file.name, open(temp_file.name, "rb")),
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0
            )
        
        # Process through LLM and generate speech
        ai_response = await process_llm(translation.text)
        audio_response = await generate_speech(ai_response)
        
        return {
            "user_input": {
                "type": "audio",
                "transcription": translation.text
            },
            "assistant_response": {
                "text": ai_response,
                "audio": audio_response
            }
        }
    except Exception as e:
        raise Exception(f"Audio processing error: {str(e)}")

async def process_text_input(text: str):
    """Process text input"""
    try:
        # Process through LLM and generate speech
        ai_response = await process_llm(text)
        audio_response = await generate_speech(ai_response)
        
        return {
            "user_input": {
                "type": "text",
                "text": text
            },
            "assistant_response": {
                "text": ai_response,
                "audio": audio_response
            }
        }
    except Exception as e:
        raise Exception(f"Text processing error: {str(e)}")

async def handler(job):
    """Main handler function for RunPod"""
    job_input = job["input"]
    job_id = job.get("id", "")
    
    try:
        input_type = job_input.get("type")
        
        if input_type == "audio":
            result = await process_audio_input(job_input["audio"])
        elif input_type == "text":
            result = await process_text_input(job_input["text"])
        else:
            return {
                "id": job_id,
                "status": "error",
                "error": f"Invalid input type: {input_type}"
            }
            
        return {
            "id": job_id,
            "status": "completed",
            "output": result
        }
        
    except Exception as e:
        return {
            "id": job_id,
            "status": "error",
            "error": str(e)
        }

# Start the RunPod handler
runpod.serverless.start({
    "handler": handler
})
