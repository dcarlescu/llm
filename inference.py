import tiktoken
import torch
from gpt import GPTModel
from config import GPT_CONFIG_SMALL, GPT_CONFIG_MEDIUM

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        #crop the input to the context size of the model
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
            
        logits = logits[:, -1, :] #focus on the last token generated
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)    
    return idx

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
#txt1 = "Every effort moves you"
txt1 = "Hello, I am"
#txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
#batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

#perform inference
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_SMALL)
model.print_model_summary()
model.eval()
idxs = generate_text_simple(model, batch, max_new_tokens=6, context_size=GPT_CONFIG_SMALL["context_length"])
generated_text = [tokenizer.decode(idx.tolist()) for idx in idxs]
print("Generated text:\n", generated_text)