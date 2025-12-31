import torch

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