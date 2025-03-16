import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio

# Create a Streamlit app object
st.sidebar.title("Lmstudio Demo")

# Model selection options
model_options = ["HuggingFaceTB/SmolLM-1.7B-Instruct", "HuggingFaceTB/SmolLM-360M-Instruct", "HuggingFaceTB/SmolLM-135M-Instruct"]
default_model_option = model_options[1]
model_selectbox = st.sidebar.selectbox("Language Model", options=model_options, index=1)

# Temperature slider
temperature_value = 0.5
temperature_slider = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, step=0.05, value=temperature_value, help="Controls the randomness of the generated text.")

# Seed slider
seed_value = 5238
seed_slider = st.sidebar.slider("Seed", min_value=0, max_value=99999, step=1, value=seed_value, help="Controls randomness of token selection.")

# Top P slider
top_p_value = 0.75
top_p_slider = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, step=0.05, value=top_p_value, help="Controls randomness and creativity.")

# Response token length slider
length_value = 256
response_token_length_slider = st.sidebar.slider("Response Token", min_value=1, max_value=99999, step=16, value=length_value, help="Maximum tokens in response.")

# Device selection
device_type = "cpu"
device_selectbox = st.sidebar.selectbox("Device Type", options=["cpu", "gpu"], index=0)

# Asynchronous text generation function
async def generate_response(prompt, device, model, top_p, temperature, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_instance = AutoModelForCausalLM.from_pretrained(model).to(device)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Use asyncio.to_thread to run the blocking model.generate in a separate thread
    loop = asyncio.get_running_loop()
    output = await loop.run_in_executor(None, lambda: model_instance.generate(inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens, do_sample=True))

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

async def main():
    st.title("Huggingface LLM App")
    user_input = st.text_input("Enter your prompt:")

    if st.button("Generate"):
        if user_input:
            with st.spinner("Generating response..."):
                response = await generate_response(
                    prompt=user_input,
                    device=device_selectbox,
                    model=model_selectbox,
                    top_p=top_p_slider,
                    temperature=temperature_slider,
                    max_new_tokens=response_token_length_slider,
                )
            st.write("Model Response:")
            st.write(response)
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    asyncio.run(main())