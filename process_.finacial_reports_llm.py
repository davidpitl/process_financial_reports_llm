from util import get_file_list, get_pdf_text_content, get_prompt, adjust_num_tokens, query_llama2, query_chatgpt3_5, \
    num_tokens


LLM = 'LLAMA2'
if LLM == 'GPT-4':
    max_tokens = 128000
elif LLM == 'LLAMA2':
    max_tokens = 2048
else:
    max_tokens = 2048


input_dir = './input_dir'


# get a list of PDF files to process
pdf_files = get_file_list(input_dir, '.pdf')


# process PDF files
for pdf_file in pdf_files:
    #extractPDF text
    text_content = get_pdf_text_content(pdf_file)

    # create llm prompt
    print(f'text_prompt: {num_tokens(text_content)}')
    text_prompt = adjust_num_tokens(text_content, max_tokens-100)
    text_prompt = get_prompt(text_prompt)
    print(f'PROMPT\n{text_prompt}\n')
    print(f'post text_prompt: {num_tokens(text_prompt)}')


    # query LLM
    if LLM == 'GPT-4':
        result_txt = query_chatgpt3_5(text_prompt, model="gpt-4-1106-preview")
    elif LLM == 'LLAMA2':
        result_txt = query_llama2(text_prompt)
    else:
        result_txt = query_llama2(text_prompt)

    print(f'result_txt: {result_txt}')

    # process reply