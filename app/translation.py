# import os, re, time
# from dotenv import load_dotenv
# import nlpcloud
# from nltk.tokenize import sent_tokenize

# # Load API key
# load_dotenv()
# client = nlpcloud.Client("nllb-200-3-3b", os.getenv("NLP_CLOUD_TOKEN"))

# # Input text
# text = """Computer networks are groups of interconnected computers and devices that can communicate electronically, sharing data, commands, and resources. These networks can be connected through various media such as cables, wireless signals, or satellites. Networks have several advantages, including resource sharing, where hardware and software resources can be shared among connected computers, reducing the need for individual purchases of licensed software. Additionally, computer networks facilitate communication between users, enabling instant messaging, video conferencing, and data transfer across distances.

# Summary: Computer networks are systems of interconnected computers and devices that enable communication, data sharing, and resource utilization. They offer benefits like resource sharing and enhanced user communication.
# """

# # Split text into paragraphs and separators (to keep blank lines)
# parts = re.split(r'(\n\s*\n)', text)

# # Collect tasks (paragraph index, sentence index, sentence)
# tasks = []
# for p_idx, part in enumerate(parts):
#     if re.fullmatch(r'\n\s*\n', part):
#         continue
#     if not part.strip():
#         continue
#     sents = sent_tokenize(part)
#     for s_idx, s in enumerate(sents):
#         tasks.append((p_idx, s_idx, s))

# # Translation function
# def translate_sentence(sent):
#     r = client.translation(sent, source="eng_Latn", target="mal_Mlym")
#     if isinstance(r, dict):
#         return r.get("translation_text") or r.get("translation") or str(r)
#     return str(r)

# # Translate sequentially with delay
# translated_map = {}
# for (p_idx, s_idx, s) in tasks:
#     translated_map[(p_idx, s_idx)] = translate_sentence(s).strip()
#     time.sleep(2)  # avoid hitting 429 rate limit

# # Reassemble text
# output_parts = []
# for p_idx, part in enumerate(parts):
#     if re.fullmatch(r'\n\s*\n', part):
#         output_parts.append(part)  # keep blank line
#         continue
#     if not part.strip():
#         output_parts.append(part)
#         continue
#     sents = sent_tokenize(part)
#     translated_sentences = [translated_map[(p_idx, i)] for i in range(len(sents))]
#     output_parts.append(" ".join(translated_sentences))

# final_output = "".join(output_parts)

# print(final_output)
# # print("\n--- Debug repr ---\n")
# # print(repr(final_output))  # to verify newlines

import os, re, time
from dotenv import load_dotenv
import nlpcloud

# Load API key
load_dotenv()
client = nlpcloud.Client("nllb-200-3-3b", os.getenv("NLP_CLOUD_TOKEN"))

# Input text
text = """Computer networks are groups of interconnected computers and other computing devices that communicate with each other electronically. These devices can share data, commands, and resources, making them more efficient and powerful than standalone computers.


One of the key benefits of computer networks is resource sharing. For example, a computer with a DVD drive can share its contents with other computers on the network, allowing them to access the data without needing to purchase a separate DVD drive. Similarly, software resources like application programs and anti-virus tools can be shared among all computers on the network, reducing the cost of purchasing licensed software for each computer.


Computer networks also facilitate communication between users. They enable users to send and receive messages, share files, and even video confer with each other in real-time, regardless of their physical location. This can significantly improve productivity and collaboration within an organization.


There are different types of computer networks, including wired and wireless networks. Wired networks use cables to connect computers, while wireless networks use electromagnetic waves, such as radio waves, to transmit data. Wireless networks are becoming increasingly popular due to their convenience and flexibility.


Guided and unguided media are also important components of computer networks. Guided media, such as coaxial cables and optical fibers, are used to transmit data between nodes in a network. Unguided media, such as wireless networks, use electromagnetic waves to transmit data between devices.


Noise is a critical issue in computer networks. It can degrade the quality of data transmission and cause errors. Noise can be caused by various factors, such as nearby radio transmitters, motors, or other cables. To minimize noise, network administrators can use noise-reducing techniques, such as shielding cables and using noise-filtering equipment.


In summary, computer networks are essential components of modern computing. They enable efficient resource sharing, real-time communication, and flexibility in data transmission. By understanding the different types of computer networks, guided and unguided media, and the importance of noise reduction, network administrators can ensure that their networks are reliable, efficient, and secure. (Sources: Section 8.1 Computer Network, Section 8.3.1 Guided Medium (Wired), Section 8.3.2 Unguided Medium (Wireless), Section 8.1.2 Some Key Terms)


[Paragraphs]


Summary: Computer networks are groups of interconnected computers and other computing devices that communicate with each other electronically. They enable efficient resource sharing, real-time communication, and flexibility in data transmission. Guided and unguided media are essential components of computer networks, and noise reduction is critical to ensure reliable and secure data transmission. (Sources: Section 8.1 Computer Network, Section 8.3.1 Guided Medium (Wired), Section 8.3.2 Unguided Medium (Wireless), Section 8.1.2 Some Key Terms)
"""

# Split into paragraphs (keep blank lines)
parts = re.split(r'(\n\s*\n)', text)

def translate_paragraph(paragraph):
    r = client.translation(paragraph, source="eng_Latn", target="mal_Mlym")
    if isinstance(r, dict):
        return r.get("translation_text") or r.get("translation") or str(r)
    return str(r)

output_parts = []
for part in parts:
    if re.fullmatch(r'\n\s*\n', part):  # blank line
        output_parts.append(part)
        continue
    if not part.strip():  # skip empty
        output_parts.append(part)
        continue
    translated = translate_paragraph(part).strip()
    output_parts.append(translated)
    time.sleep(2)  # delay between paragraphs

final_output = "".join(output_parts)

print(final_output)
