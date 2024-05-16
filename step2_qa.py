import pandas as pd
from langchain_zhipu import ZhipuAIEmbeddings, ChatZhipuAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import logging

# Set display options for pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize embeddings and other components
embeddings = ZhipuAIEmbeddings()
chroma = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = chroma.as_retriever()

chat = ChatZhipuAI(model_name="glm-4", temperature=0.5)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the prompt template
prompt = PromptTemplate.from_template("根据 {context} 请回答：{question}")

# Define the retrieval and generation chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

# Define questions and ground truths
questions = ["ChatGPT的优势有哪些"]
ground_truths = ["ChatGPT的优势有可玩性,流畅性,自我学习能力,数据标注创新,用户友好,快速响应"]

# Inference
answers = []
contexts = []

# 获取回答和上下文
for query in questions:
    logger.info(f"Processing query: {query}")
    answer = rag_chain.invoke({"question": query})
    answers.append(answer)
    relevant_docs = retriever.invoke({"query": query})
    context_list = [doc.page_content for doc in relevant_docs]
    contexts.append(context_list)

# Create data dictionary
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}

# 定义数据集特征
features = Features({
    "question": Value("string"),
    "answer": Value("string"),
    "contexts": Sequence(Value("string")),
    "ground_truth": Value("string"),
})

# Convert dict to dataset with specified features
dataset = Dataset.from_dict(data, features=features)

# Evaluate
result = evaluate(
    dataset=dataset,
    llm=chat,
    embeddings=embeddings,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    raise_exceptions=False  # 在评估过程中仅记录错误而不是中断
)

# Convert results to pandas DataFrame
df = result.to_pandas()

# Print the results
print(df)
