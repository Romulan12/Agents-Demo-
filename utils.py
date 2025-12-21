from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from typing import Tuple
from pathlib import Path

from llama_index.core.prompts import PromptTemplate
def get_doc_tools(
    document_fp: str,
    doc_name: str
) -> Tuple[QueryEngineTool, QueryEngineTool]:
    """
    Create vector and summary tools for a given document.
    
    Args:
        document_fp: Path to the document file (as string)
        doc_name: Name of the document (typically the file stem)
        
    Returns:
        Tuple of (vector_tool, summary_tool)
    """
    # Truncate doc_name to ensure tool names stay under OpenAI's 64 character limit
    # Reserve space for "_summary_tool" (13 chars) with safety margin
    max_name_length = 50
    doc_name = doc_name[:max_name_length]
    
    # Load the document
    documents = SimpleDirectoryReader(input_files=[document_fp]).load_data()
    
    # Split into nodes
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Create indexes
    summary_index = SummaryIndex(nodes=nodes)
    vector_index = VectorStoreIndex(nodes=nodes)
    
    # Create query engines
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    text_qa_template = PromptTemplate(
            """Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        If the context does not contain relevant information to answer the query, 
        respond with: "No relevant information found in this document."
        Do NOT use your general knowledge to answer.
        Query: {query_str}
        Answer: """
        )
    # vector_query_engine = vector_index.as_query_engine()
    vector_query_engine = vector_index.as_query_engine(
    text_qa_template=text_qa_template )
    # Create tools with document-specific descriptions
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        name=f"{doc_name}_summary_tool",
        description=(
            f"Useful for summarization questions related to the {doc_name} paper."
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        name=f"{doc_name}_vector_tool",
        description=(
            f"Useful for retrieving specific context from the {doc_name} paper."
        ),
    )
    
    return vector_tool, summary_tool
