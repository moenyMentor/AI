from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sys_prompt import rec_prompt
from manage_db import load_vector_db
from model import load_llm_model

db = load_vector_db()
retriever = db.as_retriever(search_kwargs={'k': 5})
prompt_content = rec_prompt 
llm = load_llm_model()

def create_rec_rag(user_input):
    rag_chain = (
        {"context": retriever, "user_input": RunnablePassthrough()}
        | prompt_content
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(user_input)
    return response