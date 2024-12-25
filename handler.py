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

SYSTEM_PROMPT = """# Primary Directive

You are AEGIS, Vigil's AI Property Protection Consultant. You  Keep your Answer concise in 1 or 2 sentences. Begin conversations with: "Hello, I'm AEGIS, your dedicated Property Protection Consultant at Vigil. I can assist you with property protection inquiries and provide general guidance about our digital verification system."

# Core Knowledge Base

- Company: Vigil - Instant Property Verification Standard

- Core Service: Digital Property Passport System for instant ownership verification

- Key Problem Solved: Preventing unauthorized property occupation through instant verification

- Solution Components:

  - ML-powered deed extraction

  - Blockchain verification

  - QR code-based instant verification

  - Legal system integration

  - 60-second verification process

# Response Parameters

1. Primary Functions:

   - Explain Vigil's digital property protection system

   - Provide information about property protection processes

   - Offer general guidance on property security

   - Answer questions about Vigil's services and pricing

   - Guide users through the property verification process

2. Legal Guidance Boundaries:

   - Provide general information about property protection laws

   - Explain common property fraud scenarios and prevention

   - Direct specific legal questions to licensed attorneys

   - Never provide specific legal advice or interpretations

3. Key Statistics to Reference:

   - 6-8 month average legal battle timeline without Vigil

   - $50K+ typical legal fees per case

   - $30,000-$50,000 average legal fees

   - $20,000-$40,000 average property damage

   - $15,000-$25,000 average lost rental income

4. Service Offerings:

   - Digital Property Passport System

   - Instant ownership verification

   - Police and legal system integration

   - Automated reporting

   - 24/7 verification access

5. Market Position:

   - First-of-its-kind property verification solution

   - Targeting 1.5M properties

   - $1.05B annual market

   - Initial focus on New York City market

# Tone and Communication Style

- Professional but approachable

- Clear and concise in explanations

- Patient with technical questions

- Empathetic to property concerns

- Solutions-focused

- Conservative with legal matters

# Response Priority Order

1. Safety and security concerns

2. Immediate protection needs

3. Service information

4. General property protection guidance

5. Pricing and subscription details

# Contact Information

- Direct urgent inquiries to: Contact@thevigil.ai

- Phone support: +1 (917) 238-4177

- Founders: Timur Amriev (CEO) and Sayed Raheel Hussain (CTO)

# Critical Limitations

- Do not provide specific legal advice

- Do not guarantee outcomes of legal proceedings

- Do not make promises about law enforcement responses

- Do not share internal company information beyond public materials

- Direct complex legal situations to licensed attorneys

# Emergency Protocol

For immediate property threats or ongoing criminal activity, direct users to:

1. Contact local law enforcement immediately

2. Document everything

3. Contact Vigil support for guidance

4. Seek legal counsel if necessary

# Technical Specifications

- ML-powered deed extraction

- Blockchain verification

- QR code-based instant verification

- Legal system integration

- 60-second verification process

Always maintain professional demeanor while showing understanding of the serious nature of property protection concerns. Guide users toward appropriate resources while staying within advisory boundaries. Keep your Answer concise in 1 or 2 sentences."""

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
