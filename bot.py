from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import uvicorn

app = FastAPI()


class GenerateRequests(BaseModel):
    prompt: str


@app.get("/generate")
def generate(request: Request, data: GenerateRequests):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    input_ids = tokenizer.encode(data.prompt, return_tensors="pt")
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    generated = model.generate(
        input_ids, attention_mask=attention_mask, max_length=1024, do_sample=True, top_p=0.95, top_k=60
    )

    return tokenizer.decode(generated.tolist()[0], skip_special_tokens=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
