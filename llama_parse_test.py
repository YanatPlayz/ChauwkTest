from dotenv import load_dotenv
load_dotenv()
from llama_parse import LlamaParse
import os

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    skip_diagonal_text=True,
    parsing_instruction="The provided documents contain many tables with addresses. If the state column is empty, that row refers to the state from the row before. Be precise in extracting information. ",
)

# sync
documents = parser.load_data("./data/Model_Career_Centres.pdf")
print(documents)
