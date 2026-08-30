from typing import List

try:
    import chromadb
except ImportError:  # pragma: no cover - optional for light deploys
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional for light deploys
    SentenceTransformer = None


# =============================================================================
# POLICY RAG AGENT
# =============================================================================

class PolicyRAGAgent:

    def __init__(self):
        self.policies = [
            {
                "id": "p1",
                "text": "Women Borrowers Scheme: 0.10% discount on standard personal and home loan interest rates for female primary applicants."
            },
            {
                "id": "p2",
                "text": "Youth Career Starter Program: Borrowers aged 21 to 29 years with salary above ₹40,000 qualify for a 0.20% rate discount."
            },
            {
                "id": "p3",
                "text": "Senior and Pre-Retirement Limit: Borrowers aged 50 years and above must conclude loan tenure before age 60."
            },
            {
                "id": "p4",
                "text": "Super Prime Credit Tier: CIBIL score equal to or above 780 gets a 0.35% interest concession."
            },
            {
                "id": "p5",
                "text": "Zero Foreclosure Fee Policy: Zero penalty for early closure after 12 successful EMIs under applicable floating and personal loan tiers."
            },
        ]

        self.client = None
        self.embedder = None
        self.collection = None

        if chromadb is not None and SentenceTransformer is not None:
            self.client = chromadb.Client()
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.collection = self.client.get_or_create_collection(name="bank_loan_policies")
            self._index_policies()

    # =========================================================================
    # INDEX POLICIES
    # =========================================================================

    def _index_policies(self):
        if self.collection is None:
            return

        if self.collection.count() > 0:
            return

        texts = [p["text"] for p in self.policies]
        embeddings = self.embedder.encode(texts).tolist()

        self.collection.add(
            ids=[p["id"] for p in self.policies],
            documents=texts,
            embeddings=embeddings,
        )

    # =========================================================================
    # RETRIEVE POLICIES
    # =========================================================================

    def retrieve(
        self,
        query: str,
        n_results: int = 3
    ) -> List[str]:

        if self.collection is None or self.embedder is None:
            query_lower = query.lower()
            matches = []
            for item in self.policies:
                if query_lower in item["text"].lower():
                    matches.append(item["text"])
            return matches[:n_results]

        q_emb = self.embedder.encode([query]).tolist()

        result = self.collection.query(
            query_embeddings=q_emb,
            n_results=n_results,
        )

        documents = result.get("documents", [])

        if not documents:
            return []

        return documents[0]