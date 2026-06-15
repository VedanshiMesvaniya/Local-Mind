from langchain_core.prompts import ChatPromptTemplate

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:""")

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an elite enterprise compliance and financial analyst. 
Analyze the provided documents and conversation history to answer the user's question.
Base your answer primarily on the provided <context> tags. Cite the source file and page number at the end.

<memory>{memory}</memory>
<context>{context}</context>
<chat_history>{chat_history}</chat_history>

Question: {question}
""")