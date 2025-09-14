# import os, re, time
# from dotenv import load_dotenv
# from openai import OpenAI  # assuming your llm_model wrapper

# load_dotenv()

# llm_model = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=os.getenv("GPT_OSS_20B")
# )

# def gen_single_ip(messages, top_p=0.75, temperature=0, max_tokens=5000, model="openai/gpt-oss-20b", **kwargs):
#     payload = {
#         "model": model,
#         "messages": messages,
#         "top_p": top_p,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#         **kwargs,
#     }
#     try:
#         completion = llm_model.chat.completions.create(**payload)
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"Error: {str(e)}"

# # Input text
#   # your multi-paragraph text
# text = """Computer networks are groups of interconnected computers and other computing devices that communicate with each other electronically. These devices can share data, commands, and resources, making them more efficient and powerful than standalone computers.


# One of the key benefits of computer networks is resource sharing. For example, a computer with a DVD drive can share its contents with other computers on the network, allowing them to access the data without needing to purchase a separate DVD drive. Similarly, software resources like application programs and anti-virus tools can be shared among all computers on the network, reducing the cost of purchasing licensed software for each computer.


# Computer networks also facilitate communication between users. They enable users to send and receive messages, share files, and even video confer with each other in real-time, regardless of their physical location. This can significantly improve productivity and collaboration within an organization.


# There are different types of computer networks, including wired and wireless networks. Wired networks use cables to connect computers, while wireless networks use electromagnetic waves, such as radio waves, to transmit data. Wireless networks are becoming increasingly popular due to their convenience and flexibility.


# Guided and unguided media are also important components of computer networks. Guided media, such as coaxial cables and optical fibers, are used to transmit data between nodes in a network. Unguided media, such as wireless networks, use electromagnetic waves to transmit data between devices.


# Noise is a critical issue in computer networks. It can degrade the quality of data transmission and cause errors. Noise can be caused by various factors, such as nearby radio transmitters, motors, or other cables. To minimize noise, network administrators can use noise-reducing techniques, such as shielding cables and using noise-filtering equipment.


# In summary, computer networks are essential components of modern computing. They enable efficient resource sharing, real-time communication, and flexibility in data transmission. By understanding the different types of computer networks, guided and unguided media, and the importance of noise reduction, network administrators can ensure that their networks are reliable, efficient, and secure. (Sources: Section 8.1 Computer Network, Section 8.3.1 Guided Medium (Wired), Section 8.3.2 Unguided Medium (Wireless), Section 8.1.2 Some Key Terms)


# [Paragraphs]


# Summary: Computer networks are groups of interconnected computers and other computing devices that communicate with each other electronically. They enable efficient resource sharing, real-time communication, and flexibility in data transmission. Guided and unguided media are essential components of computer networks, and noise reduction is critical to ensure reliable and secure data transmission. (Sources: Section 8.1 Computer Network, Section 8.3.1 Guided Medium (Wired), Section 8.3.2 Unguided Medium (Wireless), Section 8.1.2 Some Key Terms)
# """

# # Split into paragraphs (keep blank lines)
# parts = re.split(r'(\n\s*\n)', text)

# def translate_paragraph(paragraph):
#     if not paragraph.strip():
#         return paragraph
#     prompt = f"Translate the following English text to Malayalam, Do NOT add anything else. Text:\n\n{paragraph}"
#     messages = [{"role": "user", "content": prompt}]
#     translated = gen_single_ip(messages)
#     return translated.strip()

# output_parts = []
# for part in parts:
#     if re.fullmatch(r'\n\s*\n', part):  # blank line
#         output_parts.append(part)
#         continue
#     translated = translate_paragraph(part)
#     output_parts.append(translated)
#     # time.sleep(2)  # optional delay between paragraphs

# final_output = "".join(output_parts)
# print(final_output)


# # text = "Computer networks are groups of interconnected computers and other computing devices that communicate with each other electronically. These devices can share data, commands, and resources, making them more efficient and powerful than standalone computers."


# # text = "Computer networks are groups of interconnected computers and other computing devices that communicate with each other electronically. They enable efficient resource sharing, real-time communication, and flexibility in data transmission. Guided and unguided media are essential components of computer networks, and noise reduction is critical to ensure reliable and secure data transmission. (Sources: Section 8.1 Computer Network, Section 8.3.1 Guided Medium (Wired), Section 8.3.2 Unguided Medium (Wireless), Section 8.1.2 Some Key Terms)"

# # messages = [
# #     {"role": "system", "content": f"Translate the following text from English to Malayalam. Text: {text}"}
# # ]

# # response = gen_single_ip(messages)
# # print(response)

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load env vars
load_dotenv()

# Init client
llm_model = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("GPT_OSS_20B")
)

def gen_single_ip(messages, top_p=0.75, temperature=0, max_tokens=5000, model="openai/gpt-oss-20b", **kwargs):
    """Helper for single LLM call"""
    try:
        payload = {
            "model": model,
            "messages": messages,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        completion = llm_model.chat.completions.create(**payload)
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


text = """Paragraph 1: According to the Retrieved Information, the student query is about Nationalism in India. However, the relevant text in the Retrieved Information does not explicitly discuss nationalism in India. The closest text is "India and the Contemporary World" in Rank 1, Section: Discuss, Text: 2, which does not provide specific information on nationalism in India.


Paragraph 2: The text in Rank 7, Section: 6 Nationalism and Imperialism, Text: India and the Contemporary World, mentions a map celebrating the British Empire with images of tigers, elephants, and forests representing the colonies. However, this does not discuss nationalism in India.


Paragraph 3: The text in Rank 6, Section: 6 Nationalism and Imperialism, Text: India and the Contemporary World, states that anti-imperial movements developed everywhere, including India, and were nationalist in the sense that they struggled to form independent nation-states. However, this does not provide specific information on the history of nationalism in India.


Sources: Discuss,Text: 2; 6 Nationalism and Imperialism,Text: India and the Contemporary World; 6 Nationalism and Imperialism,Text: India and the Contemporary World


Summary: The Retrieved Information does not provide enough information to answer the student query about nationalism in India. The closest texts discuss anti-imperial movements in India but do not specifically address the history of nationalism in India.
"""

# Step 1: Split into paragraphs (keeping blank lines)
paragraphs = re.split(r'\n\s*\n', text)

# Step 2: Batch paragraphs
batch_size = 3
batches = [
    paragraphs[i:i + batch_size] for i in range(0, len(paragraphs), batch_size)
]

translated_output = []

# Step 3: Send each batch
for batch in batches:
    batch_text = "\n\n".join(batch)
    messages = [
        {"role": "system", "content": "You are a translator. Translate English to Malayalam. Preserve paragraph breaks exactly as input."},
        {"role": "user", "content": batch_text}
    ]
    translated = gen_single_ip(messages)
    translated_output.append(translated)

# Step 4: Join translations
final_translation = "\n\n".join(translated_output)

# Save
with open("translated_output.txt", "w", encoding="utf-8") as f:
    f.write(final_translation)

# Print result
print(final_translation, flush=True)
