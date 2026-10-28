#!/usr/bin/env python3
"""RAG chat over the archive data stored in Pinecone.

This script demonstrates the retrieval-augmented generation (RAG) flow:

    1. The user asks a question.
    2. ArchiveRetriever embeds the question with the same embedding model that
       was used during ingestion.
    3. Pinecone compares that question vector with the stored vectors and
       returns the closest archive chunks.
    4. format_docs joins those chunks into one context block.
    5. A LangChain prompt combines the context, the question, and chat history.
    6. ChatOpenAI answers using only the retrieved context.

Usage:
    python rag_chat.py "Who won the 2019 Drivers Championship?"
    python rag_chat.py --show-sources "Albert Park circuit details"
    python rag_chat.py                 # interactive chat
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "f1")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "archive")
EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBED_MODEL",
    "openai/text-embedding-3-large",
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
CHAT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
)
# How many nearby chunks to retrieve for each question.
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

SYSTEM_PROMPT = """You answer questions about the Formula 1 archive.
The archive covers seasons from 1950 through 2024. If the user asks about a
later year or about "today", explain that the archive ends in 2024 and give the
most recent answer that the context supports.
Use only the archive context below to answer. If the context does not contain
the answer, say that you do not know. When possible, cite the source file and
row range used for the answer."""


class ArchiveRetriever:
    """Find the archive chunks that are most relevant to a question.

    The embedding model here MUST match the model used by
    archive_to_pinecone.py. If the models differ, the question vector and the
    stored vectors live in different vector spaces and retrieval will be poor.
    """

    def __init__(self, index, embeddings, top_k=TOP_K, namespace=NAMESPACE):
        self.index = index
        self.embeddings = embeddings
        self.top_k = top_k
        self.namespace = namespace

    @staticmethod
    def _to_document(match):
        """Convert one Pinecone match into a LangChain Document."""
        metadata = dict(match.metadata or {})
        text = metadata.pop("text", "")
        metadata["score"] = float(match.score or 0.0)
        return Document(page_content=text, metadata=metadata)

    def _neighbor(self, vector, source, field, value):
        """Return the chunk immediately before/after a hit, if it exists.

        This helps when an answer is split across chunk boundaries. For
        example, a race result chunk may contain P4-P7 while the winner is in
        the previous chunk.
        """
        if value < 1:
            return None

        result = self.index.query(
            vector=vector,
            top_k=1,
            namespace=self.namespace,
            include_metadata=True,
            filter={
                "source": {"$eq": source},
                field: {"$eq": value},
            },
        )

        return self._to_document(result.matches[0]) if result.matches else None

    @staticmethod
    def _search_text(question):
        """Add hints when the user asks about "today" or a future season.

        The archive ends in 2024. Without this hint, a question like "who is
        the best driver today" can retrieve an old standings row. Pointing the
        search at the final 2024 data makes the answer much more useful.
        """
        lower = question.lower()
        relative_words = (
            "today",
            "current",
            "currently",
            "latest",
            "recent",
            "now",
            "2025",
            "2026",
        )

        if not any(word in lower for word in relative_words):
            return question

        if any(word in lower for word in ("winner", "won", "race", "grand prix")):
            return (
                question
                + " 2024 Abu Dhabi Grand Prix final race result winner"
            )

        return (
            question
            + " 2024 final drivers championship standings after Abu Dhabi"
        )

    def invoke(self, question):
        # Step 1: Convert the text question into a numeric vector.
        # The archive ends in 2024, so relative questions get a year hint.
        vector = self.embeddings.embed_query(self._search_text(question))

        # Step 2: Ask Pinecone for the top_k vectors closest to this vector.
        # Metadata is included because the original chunk text is stored there.
        result = self.index.query(
            vector=vector,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        # Step 3: Convert each Pinecone match into a LangChain Document.
        # A Document has page_content (the text) and metadata (source, rows).
        documents = [self._to_document(match) for match in result.matches]

        # Step 4: Add the chunks just before/after the best hits. Important
        # rows (like a race winner) can otherwise be cut off by chunk borders.
        seen = {
            (document.metadata.get("source"), document.metadata.get("row_start"))
            for document in documents
        }

        for document in documents[:3]:
            source = document.metadata.get("source")
            row_start = document.metadata.get("row_start")
            row_end = document.metadata.get("row_end")

            neighbors = [
                self._neighbor(vector, source, "row_end", row_start - 1),
                self._neighbor(vector, source, "row_start", row_end + 1),
            ]

            for neighbor in neighbors:
                if neighbor is None:
                    continue

                key = (
                    neighbor.metadata.get("source"),
                    neighbor.metadata.get("row_start"),
                )

                if key not in seen:
                    documents.append(neighbor)
                    seen.add(key)

        return documents


def format_docs(documents):
    """Turn retrieved Documents into readable context for the LLM.

    Each chunk is labelled with its source file and row range so the model can
    cite where its answer came from.
    """

    parts = []

    for number, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        row_start = document.metadata.get("row_start", "?")
        row_end = document.metadata.get("row_end", "?")
        parts.append(
            f"[{number}] {source} rows {row_start}-{row_end}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(parts)


def print_sources(documents):
    """Print which archive chunks were retrieved (for debugging/teaching)."""

    print("\nSources:")

    for number, document in enumerate(documents, start=1):
        print(
            f"  [{number}] {document.metadata.get('source')} "
            f"rows {document.metadata.get('row_start')}-"
            f"{document.metadata.get('row_end')} "
            f"(score {document.metadata.get('score', 0):.4f})"
        )


def build_components(top_k):
    """Create the retriever and the LangChain answer chain."""

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")

    if not openrouter_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    if not pinecone_key:
        raise SystemExit("PINECONE_API_KEY is not set")

    # Step 1: Connect to the Pinecone index that contains the archive vectors.
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)

    # Step 2: Use the same embedding model and dimension as ingestion.
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        api_key=openrouter_key,
        base_url=OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,
    )

    # Step 3: Wrap Pinecone in our retriever so the chat code stays simple.
    retriever = ArchiveRetriever(index, embeddings, top_k=top_k)

    # Step 4: Create the chat model that will write the final answer.
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=openrouter_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://localhost",
            "X-Title": "archive-rag-chat",
        },
    )

    # Step 5: Build the prompt template. It has three parts:
    #   - system instructions
    #   - previous chat messages (history)
    #   - the current context and question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            (
                "human",
                "Archive context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    # Step 6: LCEL pipe: prompt -> LLM -> plain string output.
    chain = prompt | llm | StrOutputParser()

    return retriever, chain


def answer_question(retriever, chain, question, history=None, show_sources=False):
    """Run one RAG turn: retrieve context, then generate an answer."""

    history = history or []

    # Step 1: Retrieve the most relevant archive chunks.
    documents = retriever.invoke(question)

    # Step 2: Put the chunks and the question into the LangChain prompt chain.
    # The chain then calls the LLM and returns plain text.
    answer = chain.invoke(
        {
            "context": format_docs(documents),
            "question": question,
            "history": history,
        }
    )

    if show_sources:
        print_sources(documents)

    return answer, documents


def run_interactive(retriever, chain, show_sources):
    """Keep asking questions in a loop and remember the conversation."""

    # Each turn's messages are appended here so follow-up questions have context.
    history = []
    print("Archive RAG chat. Type 'exit' or press Ctrl-D to quit.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        answer, _ = answer_question(
            retriever,
            chain,
            question,
            history,
            show_sources,
        )
        print(f"\nAssistant: {answer}")

        history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask questions about the archive stored in Pinecone."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask (omit for interactive chat)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of archive chunks to retrieve (default: {TOP_K})",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the retrieved source chunks",
    )
    return parser.parse_args()


def main():
    """Choose between a one-shot question, piped input, or interactive chat."""

    args = parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    # Build the retriever and LLM chain once, then reuse them.
    retriever, chain = build_components(args.top_k)

    question = " ".join(args.question).strip()

    # No question argument and stdin is piped: read the question from stdin.
    if not question and not sys.stdin.isatty():
        question = sys.stdin.read().strip()

    # One-shot mode.
    if question:
        answer, _ = answer_question(
            retriever,
            chain,
            question,
            show_sources=args.show_sources,
        )
        print(answer)
        return

    # No question supplied and running in a terminal: start a chat loop.
    run_interactive(retriever, chain, args.show_sources)


if __name__ == "__main__":
    main()
