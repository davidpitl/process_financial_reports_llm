import os
import PyPDF2
import re
import requests
import json
import sys
import tiktoken
import openai
import os

LLAMA_URL = os.getenv('LLAMA_URL')


def clean_text(text):
    if text.isspace():
        return None
    else:
        return text


def get_file_list(ruta_directorio, extension):
    archivos_json = []
    for root, _, archivos in os.walk(ruta_directorio):
        for nombre_archivo in archivos:
            if nombre_archivo.endswith(extension):
                archivos_json.append(os.path.join(root, nombre_archivo))
    return archivos_json


def get_pdf_text_content(pdf_file_path):
    output = []

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_path)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]

            def visitor_body(text, cm, tm, fontDict, fontSize):
                # y = tm[5]
                # if 50 < y < 720:

                # TODO clean text
                cleanned_text = clean_text(text)
                if cleanned_text:
                    output.append(cleanned_text)

            page.extract_text(visitor_text=visitor_body)

        output = ' '.join(output)
        output = re.sub(' +', ' ', output)

    except PyPDF2.errors.PdfReadError as e:
        print(f" Error reading PDF file: {pdf_file_path}, exception:", e)
        return None

    return output


def get_prompt(text):
    # prompt_txt = f"From the text content below, could you indicate the value of th growth of GDP at market prices:\n{text}"
    prompt_txt = f"Context: ###\n{text}\n\n From the previous text, What is the value of the growth of GDP at market prices?:\n"
    return prompt_txt


def query_llama2(prompt_txt, url=LLAMA_URL, model="/projects/llama.cpp/models/nous-hermes-llama2-13b.Q8_0.gguf",
                 temperature=0, max_tokens=2048):
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a financial report analyzer."},
            {"role": "user", "content": prompt_txt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_text = json.loads(response.text)["choices"][0]["message"]["content"].strip()
        return response_text
    except Exception as e:
        print(e)
        sys.exit()
        return None


def num_tokens(input, model_name="gpt-3.5-turbo"):
    tokenizer = tiktoken.get_encoding("cl100k_base" if model_name == "gpt-3.5-turbo" else "p50k_base")
    text_token_len = len(tokenizer.encode(input))
    return text_token_len


def adjust_num_tokens(text_prompt, max_tokens=2048):
    if num_tokens(text_prompt) <= max_tokens:
        return text_prompt

    # cut string
    tokens_actual = 0
    cut_phrases = []
    phrases = text_prompt.split('. ')

    cnt_phrases = 0
    for phrase in phrases:
        tokens_actual += num_tokens(phrase) + cnt_phrases * 2  # join char size: '. '
        # print(f'{tokens_actual}')

        if tokens_actual <= max_tokens:
            cut_phrases.append(phrase)
            cnt_phrases += 1
        else:
            break

    return '. '.join(cut_phrases)


def query_chatgpt3_5(prompt_txt, model="gpt-3.5-turbo",
                     question="",
                     max_len=2048,
                     debug=False, max_tokens=4096, stop_sequence=None,
                     ):
    # TODO max_tokens
    try:
        messages = [
            {"role": "system",
             "content": "You are a helpful conversational assitant that will maintain conversations in Spanish."},
            {"role": "user", "content": prompt_txt}
        ]

        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return None
