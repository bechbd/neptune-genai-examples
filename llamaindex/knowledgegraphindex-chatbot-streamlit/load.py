import os
from llama_index.readers.web import SimpleWebPageReader
from llama_index.graph_stores.neptune import NeptuneAnalyticsGraphStore
from llama_index.vector_stores.neptune import NeptuneAnalyticsVectorStore
from llama_index.core import (
    KnowledgeGraphIndex,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import PromptTemplate


KG_PERSIST_DIR = "storage_kg"
VSS_PERSIST_DIR = "storage_vss"
graph_identifier = "INSERT GRAPH ID"
max_triplets_per_chunk = 15
doc_lst = [
    "https://aws.amazon.com/about-aws/whats-new/2024/04/amazon-timestream-liveanalytics-fedramp-high-authorization-aws-govcloud-west-region/",
    "https://aws.amazon.com/about-aws/whats-new/2024/03/amazon-timestream-influxdb-available/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/aws-dms-amazon-timestream-target/",
    "https://aws.amazon.com/about-aws/whats-new/2024/04/elasticache-serverless-aws-canada-west-calgary-region/",
    "https://aws.amazon.com/about-aws/whats-new/2024/04/amazon-documentdb-middle-east-uae-region/",
    "https://aws.amazon.com/about-aws/whats-new/2024/04/elasticache-serverless-aws-govcloud-us-regions/",
    "https://aws.amazon.com/about-aws/whats-new/2024/03/elasticache-serverless-controls-scaling/",
    "https://aws.amazon.com/about-aws/whats-new/2024/03/elasticache-encryption-at-rest-govcloud-us-regions/",
    "https://aws.amazon.com/about-aws/whats-new/2024/03/amazon-neptune-database-aws-asia-pacific-osaka-region/",
    "https://aws.amazon.com/about-aws/whats-new/2024/03/amazon-neptune-analytics-aws-europe-london-region/",
    "https://aws.amazon.com/about-aws/whats-new/2024/03/amazon-neptune-dod-impact-level-4-5/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-documentdb-mongodb-in-place-major-version-upgrade-aws-govcloud-us-regions/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-documentdb-mongodb-compatibility-elastic-clusters-automatic-backups-snapshot-copying/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-documentdb-mongodb-elastic-clusters-readable-secondaries-start-stop-clusters/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-neptune-supports-opensearch-serverless/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-neptune-data-apis-sdk/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-neptune-io-optimized/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-documentdb-mongodb-partial-indexes/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-documentdb-vector-search-hnsw-index/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-documentdb-mongodb-maintenance-notifications/https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-documentdb-mongodb-text-search/",
    "https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-documentdb-mongodb-text-search/",
    "https://aws.amazon.com/about-aws/whats-new/2024/01/amazon-documentdb-mongodb-elastic-clusters-additional-regions/",
    "https://aws.amazon.com/about-aws/whats-new/2024/01/amazon-elasticache-redis-auto-scaling-govcloud-regions/",
    "https://aws.amazon.com/about-aws/whats-new/2024/01/amazon-documentdb-mongodb-global-clusters-govcloud/",
    "https://aws.amazon.com/about-aws/whats-new/2024/01/glide-redis-oss-redis-client-sponsored-aws-preview/",
    "https://aws.amazon.com/about-aws/whats-new/2024/01/amazon-elasticache-memcached-1-6-22/",
    "https://aws.amazon.com/about-aws/whats-new/2024/01/amazon-elasticache-additional-sizes-network-optimized-c7gn-nodes/",
    "https://aws.amazon.com/about-aws/whats-new/2023/12/amazon-documentdb-1-click-ec2-connectivity-instance/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-neptune-analytics/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/vector-search-amazon-memorydb-redis-preview/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/vector-search-amazon-documentdb/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-elasticache-serverless/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-documentdb-i-o-optimized/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-documentdb-no-code-learning-sagemaker-canvas/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-documentdb-mongodb-compatibility-aws-govcloud-us-east-region/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-neptune-aws-israel-tel-aviv-region/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-elasticache-redis-version-7-1-generally-available/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/aws-cost-management-purchase-recommendations-amazon-memory-db-reserved-nodes/",
    "https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-elasticache-network-optimized-c7gn-graviton-3-nodes/",
    "https://aws.amazon.com/about-aws/whats-new/2023/10/amazon-memorydb-graviton3-based-r7g-nodes/",
]


loader = SimpleWebPageReader()
documents = loader.load_data(doc_lst)


def create_or_load_indexes():
    graph_store = NeptuneAnalyticsGraphStore(graph_identifier=graph_identifier)
    vector_store = NeptuneAnalyticsVectorStore(
        graph_identifier=graph_identifier, embedding_dimension=1536
    )

    indexes = {
        "kg_index": load_kg_index(graph_store),
        "vss_index": load_vector_index(vector_store),
    }
    return indexes


def load_kg_index(graph_store):
    # check if kg storage already exists
    if not os.path.exists(KG_PERSIST_DIR):
        # load the documents and create the index
        kg_storage_context = StorageContext.from_defaults(graph_store=graph_store)
        print("Creating KG Index")

        text = (
            "Some text is provided below. Given the text, extract up to "
            "{max_knowledge_triplets} "
            "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
            "Triplets should be focused on entities such as AWS products, product features, and events.\n"
            "Triplets should avoid words like AWS and Amazon.\n"
            "---------------------\n"
            "Example:"
            "Text: DocumentDB (with MongoDB compatibility) now supports vector search."
            "Triplets:\n(DocumentDB, supports, vector search)\n"
            "Text: AWS announces the general availability of Amazon Neptune Analytics, a new analytics database engine\n"
            "Triplets:\n"
            "(Neptune Analytics, announces, general availability)\n"
            "---------------------\n"
            "Text: {text}\n"
            "Triplets:\n"
        )
        template: PromptTemplate = PromptTemplate(text)
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=kg_storage_context,
            max_triplets_per_chunk=max_triplets_per_chunk,
            include_embeddings=True,
            show_progress=True,
            kg_triple_extract_template=template,
        )

        # persistent storage
        kg_index.storage_context.persist(persist_dir=KG_PERSIST_DIR)
    else:
        # load the existing index
        print("Loading KG Index")
        storage_context = StorageContext.from_defaults(
            persist_dir=KG_PERSIST_DIR, graph_store=graph_store
        )
        kg_index = load_index_from_storage(storage_context)

    return kg_index


def load_vector_index(vector_store):
    # check if vss storage already exists
    if not os.path.exists(VSS_PERSIST_DIR):
        # load the documents and create the index
        vss_storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("Creating VSS Index")
        vss_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=vss_storage_context,
            max_triplets_per_chunk=max_triplets_per_chunk,
            include_embeddings=True,
            show_progress=True,
        )

        # persistent storage
        vss_index.storage_context.persist(persist_dir=VSS_PERSIST_DIR)
    else:
        # load the existing index
        print("Loading VSS Index")
        storage_context = StorageContext.from_defaults(
            persist_dir=VSS_PERSIST_DIR, vector_store=vector_store
        )
        vss_index = load_index_from_storage(storage_context)

    return vss_index
