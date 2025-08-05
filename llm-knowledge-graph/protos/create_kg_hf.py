import os
import sys

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_community.llms import SagemakerEndpoint
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
import json
import boto3
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship

from dotenv import load_dotenv
load_dotenv()

# Prompt user for PDF file path
FILE_PATH = input("Enter the path to the PDF file: ").strip()

# Content handlers for SageMaker endpoints
class LLMContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt: str, model_kwargs) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["generated_text"]

class EmbeddingContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, inputs: list[str], model_kwargs) -> bytes:
        input_str = json.dumps({"text_inputs": inputs})
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> list[list[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["embedding"]

# SageMaker LLM endpoint
llm_content_handler = LLMContentHandler()
llm = SagemakerEndpoint(
    endpoint_name=os.getenv('SAGEMAKER_LLM_ENDPOINT'),  # Your LLM endpoint name
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    model_kwargs={"temperature": 0, "max_length": 512},
    content_handler=llm_content_handler
)

# embedding_provider = OpenAIEmbeddings(
#    openai_api_key=os.getenv('OPENAI_API_KEY'),
#    model="text-embedding-ada-002"
#    )

# SageMaker embedding endpoint
embedding_content_handler = EmbeddingContentHandler()
embedding_provider = SagemakerEndpointEmbeddings(
    endpoint_name=os.getenv('SAGEMAKER_EMBEDDING_ENDPOINT'),  # Your embedding endpoint name
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    content_handler=embedding_content_handler
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

doc_transformer = LLMGraphTransformer(
    llm=llm,
    )

# Check if file exists
if not os.path.exists(FILE_PATH):
    print(f"Error: File '{FILE_PATH}' not found.")
    sys.exit(1)

# Load and split the documents
loader = PyPDFLoader(FILE_PATH)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:


    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata["page"]}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    print("Chunk embedding: ",  chunk_embedding[:10] if chunk_embedding else "None"
          )

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    print("Graph documents generated: ", len(graph_docs))

    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        for node in graph_doc.nodes:

            print("Node: ", node.id, node.type, node.properties if node.properties else ""      )

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
                    )
                )
        for relationship in graph_doc.relationships:
            print("Relationship: ", relationship.source.id, 
                  relationship.target.id, relationship.type)    
            
    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)
  

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
    }};""")