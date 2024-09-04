from langchain.schema import Document
from langchain.vectorstores import Chroma
from model import load_embedding_model

def load_text_as_documents(file_path):
    file_path = file_path
    documents = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()  # 앞뒤 공백 제거
            if line:  # 빈 줄 무시
                doc = Document(
                    page_content=line,
                    metadata={"source": file_path, "line_number": line_number}
                )
                documents.append(doc)

    print(f"총 {len(documents)}개의 문서로 로드되었습니다.")

    # 처리된 문서 확인 (예시로 처음 5개만 출력)
    for i, doc in enumerate(documents[:5], 1):
        print(f"문서 {i}:")
        print(f"내용: {doc.page_content}")
        print(f"메타데이터: {doc.metadata}")
        print("-" * 50)

    return documents

texts = load_text_as_documents("./data/saving_list.txt") + load_text_as_documents("./data/deposit_list.txt") + load_text_as_documents("./data/CMA_list.txt") + load_text_as_documents("./data/fund_list.txt")

DB_PATH = "./chroma_db"

def create_vector_db(texts, embeddings):
    db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=DB_PATH, 
        collection_name="investment_products_db"
    )
    return db

def load_vector_db():
    embeddings = load_embedding_model()
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="investment_products_db",
    )