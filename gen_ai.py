import random
import os
import asyncio
from openai import AsyncOpenAI
import google.genai as genai
from dotenv import load_dotenv
from portkey_ai import AsyncPortkey
import pandas as pd
import json
from pydantic import BaseModel

class struc(BaseModel):
    celebrities:list[str]
    brands:list[str]

load_dotenv()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_portkey = AsyncPortkey(
    cashe=True,
    provider= "openai",
    api_key=os.getenv("PORTKEY_PASSWORD"),
    virtual_key=os.getenv("OPENAI_VIRTUAL_KEY")
)
gemini_portkey =AsyncPortkey(
    cashe=True,
    provider= "google",
    api_key=os.getenv("PORTKEY_PASSWORD"),
    virtual_key=os.getenv("GEMINI_VIRTUAL_KEY")
)
# region audio
async def audio_prompt_gpt(audio_file:str):
        with open(f"audio_samples/cv-corpus-22.0-delta-2025-06-20/samples/{audio_file}","rb") as audio_file:
            response= await openai_client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            response_format="json",
            temperature=0.0,  # optional
            prompt="""This is a multilingual dataset. Identify the language and transcribe clearly.
            **Important**: Make sure the output is in json format:
            {"transcript":"....."
            }
            """)
            transcript=response.text
        prompt = f"""
                You are a transcription formatter.

                Given this transcript:
                \"\"\"{transcript}\"\"\"

                Return this info in **valid JSON format**:
                {{
                "language": "English",
                "text": "...",
                }}
                Only return raw JSON. No markdown.
                """
        response =await openai_client.chat.completions.create(
            model="gpt-4-1106-preview",  # or gpt-3.5-turbo-1106
            messages=[{"role": "user", "content": prompt}],
            temperature=0
            )
        return response.choices[0].message.content

async def audio_processing_gpt(batch_size=4):
    
    audio_samples=os.listdir("audio_samples/cv-corpus-22.0-delta-2025-06-20/samples")
    random.shuffle(audio_samples)
    os.makedirs("transcript/openai_audio", exist_ok=True)
    for i in range(0,8,batch_size):
        tasks=[audio_prompt_gpt(audioo) for audioo in audio_samples[i:i+batch_size]]
        results=await asyncio.gather(*tasks)
        for j, transcript in enumerate(results):  # assume transcripts is a list of strings
            with open(f"transcript/openai_audio/_{i+j}.json", "w", encoding="utf-8") as f:
                f.write(results[j])
#asyncio.run(audio_processing_gpt())

async def audio_prompt_gemini(audio_file:str):
        myfile =gemini_client.files.upload(file=f"audio_samples/cv-corpus-22.0-delta-2025-06-20/samples/{audio_file}")
        response =await gemini_client.aio.models.generate_content(
        model='gemini-2.5-pro',
        contents= [
        "You are a transcription engine. Transcribe the audio, detect the language, and return the output  in JSON format.",
        myfile
        ],
        config={
            
            "response_mime_type":"application/json"
        }
        )
        return response.text
async def audio_processing_gemini(batch_size=4):
    audio_samples=os.listdir("audio_samples/cv-corpus-22.0-delta-2025-06-20/samples")
    random.shuffle(audio_samples)
    os.makedirs("transcript/gemini_audio", exist_ok=True)
    for i in range(0,8,batch_size):
        tasks=[audio_prompt_gemini(audioo) for audioo in audio_samples[i:i+batch_size]]
        results=await asyncio.gather(*tasks)
        for j, transcript in enumerate(results):  # assume transcripts is a list of strings
             with open(f"transcript/gemini_audio/_{i+j}.json", "w", encoding="utf-8") as f:
                 f.write(results[j])
#asyncio.run(audio_processing_gemini())
# endregion

# region images
async def create_file(file_path):
  with open(f"image_samples/{file_path}", "rb") as file_content:
    result =await openai_client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id
async def image_prompt_gpt(batch_size=4):
    image_samples=os.listdir("image_samples")[-100:]
    os.makedirs("extracted-info/openai_images",exist_ok=True)
    prompt_text="""
    you are given 4 images:
    first:Extract high-level image captions/descriptions.
    second:Identify and extract the names of celebrities and/or brands present in each image.
    make sure to return the output as a list of 4 json objects(one for every image):
    [
    {
    "celebrities": ["Name1", "Name2"],
    "brands": ["Brand1", "Brand2"]
    },
    ........
    ]
    """
    for i in range(0,8,batch_size):
        response = await openai_client.responses.create(
        model="gpt-4.1",
        input=[
                {
                    "role": "user",
                    "content":[
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image","file_id":await create_file(image_samples[i+0]),"detail":"high"},
                        {"type":"input_image","file_id":await create_file(image_samples[i+1]),"detail":"high"},
                        {"type":"input_image","file_id":await create_file(image_samples[i+2]),"detail":"high"},
                        {"type":"input_image","file_id":await create_file(image_samples[i+2]),"detail":"high"}
                    ]
                }
            ]
        )
        results=json.loads(response.output_text)
        for j,info in enumerate(results):
            with open(f"extracted-info/openai_images/{i+j}info.json","w",encoding="utf-8")as f:
                json.dump(info,f,indent=2)
#asyncio.run(image_prompt_gpt())

async def gemeni_prompt_image(image_path:str):
    my_file = gemini_client.files.upload(file=f'image_samples/{image_path}')
    response =await gemini_client.aio.models.generate_content(
    model="gemini-2.5-flash",
    contents=[my_file, """
                    you are given an image:
                    first:Extract high-level image captions/descriptions.
                    second:Identify and extract the names of celebrities and/or brands present in each image.
                    make sure to return the output in raw valid json format no markdown, no triple quotes:
                    {
                    "celebrities": ["Name1", "Name2"],
                    "brands": ["Brand1", "Brand2"]
                    }
                    """]
    )
    return response.text
async def image_processing_gemini(batch_size=4):
    os.makedirs("extracted-info/gemeni_images",exist_ok=True)
    image_sample=os.listdir("image_samples")
    for i in range(0,8,batch_size):
        tasks=[gemeni_prompt_image(image) for image in image_sample[i:i+batch_size]]
        results=await asyncio.gather(*tasks)
        for j,info in enumerate(results):
            with open(f"extracted-info/gemeni_images/info{i+j}.json","w",encoding="utf-8")as f:
                f.write(info)
#asyncio.run(image_processing_gemini())
# endregion

#region comments
df=pd.read_csv("comments.csv")
comments=df["Comment"]
async def portkey_gpt_prompt(comment:str):
    response=await openai_portkey.chat.completions.create(
        temperature=0,
        cashe=True,
        route="openai",
        model="gpt-4o",
        messages=[{"role":"user","content":f"""you are given a comment i want you to summarize it and reason if the sentiment is positive or ngative or neutral"
        make sure to return the outout in raw json format no markdown, no text, no triple quotes:
        {{
            summary:......
            sentiment:......
        }}
         the comment is below:          
         "{comment}"  """} ]
    )
    return response.choices[0].message["content"]
async def portkey_gpt_processing(batch_size=5):
    os.makedirs("generated-responses/portkey_gpt",exist_ok=True)
    for i in range(0,10,batch_size):
       tasks=[portkey_gpt_prompt(comment) for comment in comments[i:i+batch_size]]
       results=await asyncio.gather(*tasks)
       for j,comm in enumerate(results):
           with open(f"generated-responses/portkey_gpt/{i+j}_comm.json","w")as f:
               f.write(comm)
#asyncio.run(portkey_gpt_processing())

async def portkey_gemini_prompt(comment:str):
    response=await gemini_portkey.chat.completions.create(
        route="gemini",
        cashe=True,
        model="gemini-2.5-flash",
        messages=[{"role":"user","content":f"""you are given a comment i want you to summarize it and reason if the sentiment is positive or ngative or neutral"
        make sure to return the outout in raw valid json format no markdown, no triple quotes:
        {{
            summary:......
            sentiment:......
        }}
         the comment is below:          
        \"\"\"{comment}\"\"\"
        """} ],
        config={
        "response_mime_type": "application/json"
        }
        )
    return (response.choices[0].message["content"])
async def portkey_gemini_processing(batch_size=5):
    os.makedirs("generated-responses/portket_gemini",exist_ok=True)
    for i in range(0,10,batch_size):
        tasks=[portkey_gemini_prompt(comment) for comment in comments[i:i+batch_size]]
        results=await asyncio.gather(*tasks)
        for j,comm in enumerate(results):
            with open(f"generated-responses/portket_gemini/{i+j}_comm.json","w")as f:
                f.write(comm)
#asyncio.run(portkey_gemini_processing())
#endregion

















































