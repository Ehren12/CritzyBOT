from dotenv import load_dotenv
import os

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from helpers import in_cache

from fastapi import FastAPI
from pydantic import BaseModel
import redis


load_dotenv()

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill", local_files_only=True)
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill", local_files_only=True)
class UserMSGRequest(BaseModel):
    message: str
    
r = redis.Redis(
  host='eu2-suitable-cod-32440.upstash.io',
  port=32440,
  password=os.getenv('UPSTASH_REDIS_PWD'),
  ssl=True
)

app = FastAPI()

@app.post("/generate-message")
async def root(utterance: UserMSGRequest):
    key_list = r.keys()
    is_in_cache = in_cache(utterance.message, key_list)
    if is_in_cache.status:
        print("From Cache!")
        return r.get(is_in_cache.closest_match)
    inputs = tokenizer(utterance.message, return_tensors = "pt")
    results = model.generate(**inputs)
    response = tokenizer.decode(results[0])
    r.set(utterance.message, response, 6000)
    return response