## Libraries to be used --------------------------
import torch
import string
import streamlit as st
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


@st.cache(allow_output_mutation=True)
def get_model():
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.eval()
    return model

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>',
                                                    eos_token='</s>',
                                                    unk_token='<unk>',
                                                    pad_token='<pad>',
                                                    mask_token='<mask>')


## Page title and favicon ------------------------
st.set_page_config(page_title = "AI단락생성기",
                   page_icon = "https://em-content.zobj.net/source/skype/289/robot_1f916.png")

## Page content logo and header ------------------
st.text("")
st.image("https://em-content.zobj.net/source/skype/289/robot_1f916.png", width=125,)


## Title -----------------------------------------
st.title("AI Paragraph Generator")

## Header/Subheader ------------------------------
st.subheader("Generate a paragraph with just one sentence.")


## Text input
default_text = "인공지능을 잘 사용하기 위해서는"

N_SENT = 3

model = get_model()
#st.title("Text-generator")

text = st.text_input("시작할 문장이나 문구를 입력하세요👇", value=default_text)
st.write(text)
punct = ('!', '?', '.')


## Buttons ---------------------------------------
if st.button("Submit") and text:
    st.markdown("## Predict")
    with st.spinner('processing..'):
        print(f'input > {text}') 
        input_ids = tokenizer(text)['input_ids']
        gen_ids = model.generate(torch.tensor([input_ids]),
                                    max_length=500,
                                    repetition_penalty=2.0)
        generated = tokenizer.decode(gen_ids[0,:].tolist()).strip()
        if generated != '' and generated[-1] not in punct:
            for i in reversed(range(len(generated))):
                if generated[i] in punct:
                    break
            generated = generated[:(i+1)]
        print(f'KoGPT > {generated}')
    st.write(generated)

