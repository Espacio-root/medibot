from transformers import AutoTokenizer, AutoModelForCausalLM

def inference_model(model, tokenizer, prompt, max_length=200):

    for _ in range(max_length):
        tokens = tokenizer(prompt, return_tensors="pt")
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        output = model.generate(
            **tokens, do_sample=True, max_length=1)
        output = tokenizer.decode(output[0])

        prompt += output
        yield output

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("llSourcell/medllama2_7b")
    model = AutoModelForCausalLM.from_pretrained("llSourcell/medllama2_7b")

    return model, tokenizer

load_model()
