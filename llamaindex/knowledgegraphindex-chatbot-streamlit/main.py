# main.py

import os
import streamlit as st
from llama_index.core import (
    KnowledgeGraphIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.settings import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding, Models
from llama_index.readers.web import SimpleWebPageReader
from llama_index.graph_stores.neptune import NeptuneAnalyticsGraphStore

# ------------------------------------------------------------------------
# LlamaIndex - Amazon Bedrock

llm = Bedrock(model="anthropic.claude-v2")
embed_model = BedrockEmbedding(model="amazon.titan-embed-text-v1")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024
graph_identifier = "<INSERT GRAPH ID>"

# ------------------------------------------------------------------------
# Streamlit

# Page title
st.set_page_config(page_title="Neptune-Llama Q&A for all your Neptune questions 📂")


# Clear Chat History function
def clear_screen():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


with st.sidebar:
    st.title("Neptune-Llama 🦙")
    st.image("llamaposeidon.png", use_column_width=True)
    st.subheader("Q&A for Neptune")
    st.markdown(
        """[Amazon Neptune](https://aws.amazon.com/neptune/) - The easiest way to build and scale
        generative AI applications with graphs"""
    )
    st.divider()
    streaming_on = st.toggle("Streaming")
    st.button("Clear Screen", on_click=clear_screen)


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing your data. This may take a while..."):
        PERSIST_DIR = "storage_website"
        graph_store = NeptuneAnalyticsGraphStore(graph_identifier=graph_identifier)
        # check if storage already exists
        if not os.path.exists(PERSIST_DIR):
            # load the documents and create the index
            loader = SimpleWebPageReader()
            documents = loader.load_data(
                [
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/doc-history.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/notebooks.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/create-notebook-cfn.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/create-notebook-console.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/create-graph-using-console.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/loading-data.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/loading-data-formats.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/iam-roles-for-loading.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/neptune-load-batch.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/bulk-loading.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/algorithms.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/bfs-standard.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/bfs-parents.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/bfs-levels.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/sssp-bellmanFord.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/sssp-bellmanFord-parents.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/sssp-deltaStepping.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/sssp-deltaStepping-parents.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/topk-sssp.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/degree.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/degree-mutate.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/page-rank.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/page-rank-mutate.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/closeness-centrality.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/closeness-centrality-mutate.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/common-neighbors.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/total-neighbors.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/jaccard-similarity.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/overlap-similarity.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/wcc.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/wcc-mutate.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/label-propagation.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/label-propagation-mutate.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/scc.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/scc-mutate.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vector-index.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vss-algorithms.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vectors-distance.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vectors-get.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vectors-topKByEmbedding.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vectors-topKByNode.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vectors-upsert.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/vectors-remove.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/analytics-limits.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-APIs.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-APIs-execute-query.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-APIs-list-queries.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-APIs-get-query.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-APIs-cancel-query.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-APIs-graph-summary.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-APIs-IAM-role-mappings.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-plan-cache.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-explain.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-statistics.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-exceptions.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-openCypher-data-model.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-openCypher-standards-compliance.html",
                    "https://docs.aws.amazon.com/neptune-analytics/latest/userguide/query-isolation-level.html",
                ]
            )
            graph_store = NeptuneAnalyticsGraphStore(graph_identifier=graph_identifier)
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
            index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                max_triplets_per_chunk=15,
                include_embeddings=True,
                show_progress=True,
            )
            # persistent storage
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(
                persist_dir=PERSIST_DIR, graph_store=graph_store
            )
            index = load_index_from_storage(storage_context)
        return index


# Create Index
index = load_data()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input - User Prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if streaming_on:
        # Query Engine - Streaming
        query_engine = index.as_query_engine(streaming=True)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            streaming_response = query_engine.query(prompt)
            for chunk in streaming_response.response_gen:
                full_response += chunk
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

    else:
        # Query Engine - Query
        query_engine = index.as_query_engine()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_engine.query(prompt)
                st.write(response.response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.response}
                )
