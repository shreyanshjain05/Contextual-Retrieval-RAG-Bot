import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()


class Chatbot:
    def __init__(self, data_directory="data",
                 persist_directory="vector_database",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.document_metadata = {}

        self.content_categories = {
            'geography': r'geography|location|border|mountain|river|climate|state|territory|area|region',
            'history': r'history|ancient|medieval|colonial|independence|empire|civilization|dynasty|war',
            'politics': r'government|political|parliament|president|minister|election|democracy|constitution',
            'culture': r'culture|festival|religion|language|art|dance|music|literature|tradition|custom',
            'economy': r'economy|gdp|industry|agriculture|business|trade|company|economic|finance|market',
            'tourism': r'tourism|tourist|destination|temple|monument|heritage|travel|visit|attraction',
            'education': r'education|school|college|university|iit|iim|study|academic|learning',
            'sports': r'sport|cricket|hockey|football|olympic|game|player|match|tournament',
            'food': r'food|cuisine|dish|recipe|cooking|restaurant|spice|sweet|meal|eating',
            'general': r'basic|information|fact|overview|introduction|about|india|bharat'
        }

    def load_document(self, file_path: str) -> Optional[Document]:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if documents:
                combined_content = "\n\n".join([doc.page_content for doc in documents])
                if not combined_content.strip():
                    print("Error: Document has no content")
                    return None

                combined_doc = Document(
                    page_content=combined_content,
                    metadata={
                        'source': file_path,
                        'file_name': os.path.basename(file_path),
                        'processed_date': datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                    }
                )
                self.document_metadata = combined_doc.metadata
                print(f"Document loaded: {file_path} with {len(combined_content)} characters")
                return combined_doc
            else:
                print("No content found in PDF.")
                return None
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return None

    def extract_contextual_metadata(self, content: str) -> Dict[str, Any]:
        content_lower = content.lower()
        categories_found = []
        for category, pattern in self.content_categories.items():
            if re.search(pattern, content_lower):
                categories_found.append(category)

        # Convert list to comma-separated string for ChromaDB compatibility
        categories_str = ",".join(categories_found) if categories_found else "general"

        return {
            'categories_present': categories_str,
            'content_length': len(content),
            'sections_count': len(content.split('\n\n'))
        }

    def identify_chunk_category(self, content: str) -> str:
        content_lower = content.lower()
        category_score = {}
        for category, pattern in self.content_categories.items():
            matches = len(re.findall(pattern, content_lower))
            if matches > 0:
                category_score[category] = matches
        return max(category_score, key=category_score.get) if category_score else 'general'

    def create_chunk_context(self, category: str) -> str:
        category_descriptions = {
            'geography': "Geographic information such as rivers, mountains, states, and climate conditions.",
            'history': "Historical background including ancient events, empires, and independence movements.",
            'politics': "Details about government systems, elections, political figures, and laws.",
            'culture': "Cultural aspects such as festivals, languages, traditions, and art forms.",
            'economy': "Economic indicators, industries, trade, GDP, and financial systems.",
            'tourism': "Tourist destinations, historical sites, and attractions worth visiting.",
            'education': "Educational institutions, literacy, academic performance, and systems.",
            'sports': "Information about popular sports, athletes, events, and competitions.",
            'food': "Descriptions of cuisine, dishes, ingredients, and regional food habits.",
            'general': "General information providing a basic overview or introduction."
        }
        return category_descriptions.get(category, "General information without a specific category.")

    def split_and_store_document(self, document: Document) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents([document])

        for chunk in chunks:
            # Extract contextual metadata
            contextual_metadata = self.extract_contextual_metadata(chunk.page_content)
            chunk.metadata.update(contextual_metadata)

            # Add category and context
            category = self.identify_chunk_category(chunk.page_content)
            chunk.metadata["category"] = category
            chunk.metadata["context"] = self.create_chunk_context(category)

        try:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Stored {len(chunks)} chunks into vector store at '{self.persist_directory}'.")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            # If there are still metadata issues, filter out complex metadata
            from langchain_community.vectorstores.utils import filter_complex_metadata
            filtered_chunks = filter_complex_metadata(chunks)
            self.vector_store = Chroma.from_documents(
                documents=filtered_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Stored {len(filtered_chunks)} chunks into vector store with filtered metadata.")

    def get_categories_from_string(self, categories_str: str) -> list:
        """Helper method to convert comma-separated string back to list"""
        if not categories_str or categories_str == "general":
            return ["general"]
        return [cat.strip() for cat in categories_str.split(",")]

    def search_information(self, query: str, k: int = 4) -> List[Document]:
        """Search for relevant information"""
        if not self.vector_store:
            print("Error: Knowledge base not initialized")
            return []

        try:
            # Enhanced query for better retrieval
            enhanced_query = f"institutional information query: {query}"
            results = self.vector_store.similarity_search(enhanced_query, k=k)
            print(f"Found {len(results)} relevant documents for query: {query}")
            return results

        except Exception as e:
            print(f"Error searching information: {str(e)}")
            return []

    def get_response(self, query: str) -> str:
        """Generate comprehensive response to user queries"""
        # Search for relevant information
        search_results = self.search_information(query, k=4)

        if not search_results:
            return "I couldn't find specific information about that in our database. Could you please rephrase your question or ask about something more specific?"

        # Prepare context
        context_parts = []
        categories_found = set()

        for i, result in enumerate(search_results, 1):
            category = result.metadata.get('primary_category', 'general')
            categories_found.add(category)
            context_parts.append(f"Information {i}:\n{result.page_content}\n")

        context = "\n".join(context_parts)
        institution_name = search_results[0].metadata.get('institution_name', 'the institution')

        # Generate response using LLM
        try:
            system_prompt = f"""You are an intelligent chatbot assistant for {institution_name}. You help students, parents, and prospective students get accurate information about the institution.

            Instructions:
            - Provide comprehensive, accurate answers based ONLY on the provided institutional context
            - Be helpful, professional, and friendly
            - Structure your responses clearly with bullet points or sections when appropriate
            - If specific information isn't available in the context, clearly state that
            - Always mention relevant contact information when available
            - Suggest related topics the user might be interested in
            - Keep responses concise but complete"""

            user_prompt = f"""Query: "{query}"

            Institutional Context:
            {context}

            Please provide a helpful and accurate response based on the institutional information provided."""

            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=800,
                temperature=0.3
            )

            answer = response.choices[0].message.content

            # Add helpful footer
            if len(categories_found) > 1:
                answer += f"\n\nNote: This information covers: {', '.join(sorted(categories_found))}"

            return answer

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm having trouble processing your request right now. Please try again or contact the institution directly."


def main():
    """Run the institutional chatbot"""
    print("Institutional Chatbot Assistant")
    print("-" * 40)

    # Initialize chatbot
    chatbot = Chatbot()

    # Load document
    pdf_path = os.path.join("data", "knowledge_base.pdf")
    print(f"Loading document from: {pdf_path}")

    doc = chatbot.load_document(pdf_path)

    if not doc:
        print("Failed to load document. Please check the file path and format.")
        return

    print("Document loaded successfully.")
    chatbot.split_and_store_document(doc)
    print("Document processed and vector store created.")

    print("\nChat interface ready. Type 'help' for commands.")

    while True:
        try:
            user_input = input("\nYou: ").strip().lower()

            if user_input in ['quit', 'exit', 'bye', 'q']:
                print("Goodbye!")
                break

            if user_input == 'help':
                print("Commands:\n - Ask a question\n - 'quit', 'exit' to leave\n - 'clear' to clear screen")
                continue

            if user_input == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            if not user_input:
                print("Enter a question or 'help'.")
                continue

            response = chatbot.get_response(user_input)
            print(f"Assistant: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()