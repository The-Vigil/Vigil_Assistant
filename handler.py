import runpod
import base64
from groq import Groq
from openai import OpenAI
import os
import asyncio
import tempfile

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
    """Generate text-to-speech with streaming"""
    try:
        # Stream the TTS response directly to base64
        tts_response = await asyncio.to_thread(
            openai_client.audio.speech.create,
            model="tts-1",
            voice="onyx",
            input=text,
            response_format="mp3"  # Explicitly set format
        )
        
        # Efficiently stream and encode to base64
        audio_chunks = []
        async for chunk in tts_response:
            audio_chunks.append(chunk)
        return base64.b64encode(b''.join(audio_chunks)).decode()
    except Exception as e:
        raise Exception(f"TTS generation error: {str(e)}")

async def process_audio_input(audio_base64: str):
    """Process audio input with efficient file handling"""
    try:
        # Decode audio directly to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Use context manager for automatic cleanup
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            # Stream transcription
            translation = await asyncio.to_thread(
                groq_client.audio.translations.create,
                file=temp_file.name,
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0
            )
        
        # Parallel processing of LLM and TTS
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
    """Process text input with parallel processing"""
    try:
        # Process LLM and generate speech in parallel
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
    """Main handler function for RunPod with improved response formatting"""
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