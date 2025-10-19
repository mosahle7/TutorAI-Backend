from langchain_core.messages import SystemMessage, HumanMessage

def gen_questions(llm_model, topic, norm_topic, retrieved_data, num_questions):
    sys_msg = f"""You are TutorAI, an AI assistant that helps users by generating relevant questions based ONLY on the provided topic and context.

Follow these rules strictly:
RULES:
    - Generate exactly most relevant {num_questions} questions.
    - If Retrived Information is truly insufficient to generate {num_questions} questions, generate fewer questions and at the end tell: "I could generate only 'actual number' questions on the topic: {topic} due to insufficient information in the document.". 
    - If Retrieved Information is truly irrelevant to the topic, respond with "I couldn't generate any questions as the uploaded document does not provide any information on: {topic}.".
    - Generate questions using ONLY "Retrieved Information".
    - Never use your pre-existing knowledge even if that helps.
    - Ensure questions are clear and relevant.
    - Do NOT mention "Retrieved Information", rules, or reasoning steps.
    - Do NOT repeat questions.
    - Number ONLY the questions starting from 1.
    - Do NOT number exception statements.
    - Mention the source for each question in parentheses after the question.
    - Just output the questions with sources, nothing else.

RULES ON SOURCES:
  - Mention only the sources in Retrieved Information that are used to generate the question.

  EXAMPLE:
    If this is there in Retrieved Information and you use it to generate a question:
      Section: 8.1 Computer Network
      Text:
      Computer network is a group of computers and other computing hardware devices (such as printers, scanners, modems, CD drives, etc.) connected to each other electronically through a communication medium. They can communicate with each other, exchange commands, share data, hardware, and other resources. Computers on a network may be linked through cables, telephone lines, radio waves, satellites, or infrared light beams.

    Then while citing the source, you should write:
        (Source: Section: 8.1 Computer Network)

RESPONSE FORMAT:

1. Question 1? (Source: source1)
[blank line]
2. Question 2? (Source: source2)
[blank line]
......

- Retrieved information are ordered by relevance, most relevant first.

Retrieved Information: 
{retrieved_data}
"""
    messages = [SystemMessage(content=sys_msg)]
    messages.append(HumanMessage(content=f'''
Topic: {topic}
Corrected Topic (for matching with documents): {norm_topic}
'''))

    response = llm_model.invoke(messages)
    print("Generated Questions:\n", response.content)
    return response.content