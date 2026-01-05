def build_mcq_prompt(context, question, choices):
    return f"""Based on the following document content, please answer the multiple choice question.

Document Content:
{context}

Question: {question}

Options:
A. {choices['A']}
B. {choices['B']}
C. {choices['C']}
D. {choices['D']}

Please carefully analyze the document content and select the correct answer. Respond in JSON format with the following structure:
{{
    "answer": "A/B/C/D"
}}"""

def truncate_chunks_by_topk(tokenizer, relevant_chunks):
    truncated_chunks = []
    for chunk in relevant_chunks:
        tokens = tokenizer.encode(chunk)
        if len(tokens) > 4000:
            print(f"Chunk is too long!! Truncated from {len(tokens)} to 4000 tokens.")
            tokens = tokens[:4000]
            chunk = tokenizer.decode(tokens, skip_special_tokens=True)
        truncated_chunks.append(chunk)
    context = "\n\n".join(truncated_chunks)
    return context

