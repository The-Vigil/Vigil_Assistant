import runpod
import base64
from groq import Groq
from openai import OpenAI
import os
import time
import io

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# Initialize clients
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """ # Primary Directive

You are helpful Vigil's AI assistant named AEGIS and Vigil's AI Property Protection Consultant. Respond to user inquiries in **1-2 concise sentences**, avoiding repetition or unnecessary information"

# Core Knowledge Base

- **Company**: Vigil - Instant Property Verification Standard
- **Core Service**: Digital Property Passport System for instant ownership verification
- **Primary Goal**: Prevent unauthorized property occupation via instant verification.

# Features Overview

1. **Verification Tools**:
   - ML-powered deed extraction
   - Blockchain authentication
   - QR code-based verification
   - Legal system integration
   - 60-second verification process

2. **Service Benefits**:
   - Reduced legal disputes (6-8 months on average without Vigil)
   - Savings on typical legal fees ($50K+ per case)
   - Prevention of property damage ($20K-$40K on average)

# Communication Guidelines

- **Tone**: Professional, approachable, and empathetic.
- **Response Style**: Focused, clear, and concise. Limit technical jargon unless specifically requested.
- **Legal Boundaries**: 
  - Provide general guidance on property security and fraud prevention.
  - Avoid legal interpretations or guarantees.
  - Direct complex legal queries to attorneys or law enforcement.

# Critical Response Priorities

1. Address safety and security concerns first.
2. Offer actionable guidance on property verification.
3. Provide concise information about Vigil's services or pricing.

# Emergency Protocols

For immediate property threats:
1. Advise contacting law enforcement.
2. Recommend documenting the situation.
3. Suggest contacting Vigil support: Contact@thevigil.ai | +1 (917) 238-4177.
4. Encourage consulting a legal professional.

# Key Limitations

- Do not offer legal advice or guarantee outcomes.
- Avoid repetitive openings after the first interaction.

Your primary goal is to resolve user queries quickly and effectively, while remaining professional and maintaining boundaries."""



def process_in_chunks(file_path):
    """Process file in chunks for better memory usage"""
    audio_chunks = []
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            audio_chunks.append(base64.b64encode(chunk).decode())
    return "".join(audio_chunks)

async def async_handler(job):
    try:
        start_time = time.time()
        print("\n=== New Request Started ===")
        
        # Get input from job
        input_type = job["input"]["type"]
        
        if input_type == "text":
            text_input = job["input"]["text"]
            print("Processing text input")
        else:
            print("Processing audio input...")
            audio_start = time.time()
            
            # Process incoming audio in chunks
            audio_base64 = job["input"]["audio"]
            audio_bytes = base64.b64decode(audio_base64)
            
            temp_filename = "/tmp/temp_recording.wav"
            with open(temp_filename, "wb") as f:
                # Write in chunks
                buffer = io.BytesIO(audio_bytes)
                while True:
                    chunk = buffer.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                
            with open(temp_filename, "rb") as file:
                translation = groq_client.audio.translations.create(
                    file=(temp_filename, file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    temperature=0.0
                )
            text_input = translation.text
            print(f"Audio transcription took {time.time() - audio_start:.2f}s")
        
        # LLM Response
        llm_start = time.time()
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_input}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=2048
        )
        ai_response = chat_completion.choices[0].message.content
        print(f"LLM response took {time.time() - llm_start:.2f}s")
        
        # TTS Generation with chunked processing
        tts_start = time.time()
        print("Starting TTS generation...")
        
        output_path = "/tmp/response.wav"
        
        # Generate TTS using OpenAI
        tts_response = openai_client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=ai_response
        )
        
        # Save to file using chunks
        with open(output_path, "wb") as f:
            buffer = io.BytesIO()
            for chunk in tts_response.iter_bytes(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        
        # Process output audio in chunks
        audio_base64 = process_in_chunks(output_path)
        print(f"TTS generation took {time.time() - tts_start:.2f}s")
        
        # Cleanup
        if os.path.exists("/tmp/temp_recording.wav"):
            os.remove("/tmp/temp_recording.wav")
        if os.path.exists(output_path):
            os.remove(output_path)
            
        print(f"Total request time: {time.time() - start_time:.2f}s")
        
        return {
            "user_input": {
                "type": input_type, 
                "text": text_input
            },
            "assistant_response": {
                "text": ai_response, 
                "audio": audio_base64
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}

print("Starting server...")
print("Server ready!")

runpod.serverless.start({
    "handler": async_handler
})
