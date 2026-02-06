from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever):
    llm = GoogleGenerativeAI(model="gemini-flash-latest")

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    return parallel | prompt | llm | StrOutputParser()
