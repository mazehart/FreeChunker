# Modified from https://github.com/joaodsmarques/LumberChunker/
import time
import re
import pandas as pd
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/share/home/ecnuzwx/UnifiedRAG/cache/models--Qwen--Qwen3-8B")


# Count_Words idea is to approximate the number of tokens in the sentence.
def count_words(input_string):
    return len(tokenizer.encode(input_string))
    

# Function to add IDs to each Dataframe Row
def add_ids(row):
    global current_id
    # Add ID to the chunk
    row['Chunk'] = f'ID {current_id}: {row["Chunk"]}'
    current_id += 1
    return row

system_prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""

def LLM_prompt(user_prompt, api_name, api_configure):
    """Call LLM model (Adapted for local_model support)"""
    match api_name:
        case 'local_model':
            llm = api_configure['llm']
            response = llm.chat(user_prompt)
            if isinstance(response, str):
                return response
            if isinstance(response, (list, tuple)):
                r0 = response[0]
                if isinstance(r0, str):
                    return r0
                if hasattr(r0, "text"):
                    return r0.text
                return str(r0)
            if hasattr(response, "text"):
                return response.text
            return str(response)
        case _:
            raise ValueError("This model has not yet been implemented.")

def split_text_by_punctuation(text): 
    full_segments = sent_tokenize(text)
    return full_segments

def lumberchunker(api_name, api_configure, text, timeout_seconds=None):
    start_time = time.time() 

    id_sentence_list = split_text_by_punctuation(text)
    id_chunks = pd.DataFrame(id_sentence_list, columns=['Chunk'])  

    # Initialize a global variable for current_id and Apply the function along the rows of the DataFrame
    global current_id
    current_id = 0
    id_chunks = id_chunks.apply(add_ids, axis=1) # Put ID: Prefix before each paragraph

    chunk_number = 0
    new_id_list = []
    word_count_aux = []
    
    while chunk_number < len(id_chunks)-5:
        if timeout_seconds is not None and (time.time() - start_time) > timeout_seconds:
            break

        word_count = 0
        i = 0             
        while word_count < 550 and i+chunk_number<len(id_chunks)-1:
            i += 1
            final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
            word_count = count_words(final_document)
        
        if(i == 1):
            final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
        else:
            final_document = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i-1 + chunk_number))
        
        question = f"\nDocument:\n{final_document}"

        word_count = count_words(final_document)    
        word_count_aux.append(word_count)
        chunk_number = chunk_number + i-1
    
        prompt = system_prompt + question
        
        try:
            gpt_output = LLM_prompt(prompt, api_name, api_configure)
        except Exception as e:
            print(f"⚠️ LLM call failed (treated as timeout interruption): {e}")
            break

        # For books where there is dubious content, Gemini refuses to run the prompt and returns mistake.
        if gpt_output == "content_flag_increment":
            chunk_number = chunk_number + 1

        else:
            pattern = r"Answer: ID \w+"
            match = re.search(pattern, gpt_output)

            if match == None:
                print("repeat this one")
            else:
                gpt_output1 = match.group(0)
                pattern = r'\d+'
                match = re.search(pattern, gpt_output1)
                chunk_number = int(match.group())
                new_id_list.append(chunk_number)
                if(new_id_list[-1] == chunk_number):
                    chunk_number = chunk_number + 1

    # Add the last chunk to the list
    new_id_list.append(len(id_chunks))

    # Remove IDs as they no longer make sense here.
    id_chunks['Chunk'] = id_chunks['Chunk'].str.replace(r'^ID \d+:\s*', '', regex=True)

    # Create final dataframe from chunks
    new_final_chunks = []
    for i in range(len(new_id_list)):
        # Calculate the start and end indices of each chunk
        start_idx = new_id_list[i-1] if i > 0 else 0
        end_idx = new_id_list[i]
        new_final_chunks.append('\n'.join(id_chunks.iloc[start_idx: end_idx, 0]))
    
    end_time = time.time()  
    execution_time = end_time - start_time  
    print(f"The program execution time is: {execution_time} seconds.")
    
    return new_final_chunks

class LumberChunking:
    """Lumber Chunking Class"""
    
    def __init__(self, llm=None):
        self.llm = llm

    def chunk(self, text, timeout_seconds=None):
        """Execute chunking operation"""
        if self.llm is None:
            print(f"[DEBUG] LumberChunking.chunk: Error - No LLM model provided")
            return None
        
        # Create adapter interface, use local model
        api_configure = {'llm': self.llm}
        result = lumberchunker('local_model', api_configure, text, timeout_seconds=timeout_seconds)
        return result
