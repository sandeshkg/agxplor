import streamlit as st
import os
from typing import Annotated, Dict, TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
import sqlite3
import time
import json

load_dotenv()

class IterationLimitError(Exception):
    pass

class TimeoutError(Exception):
    pass

def load_policy_documents():
    """Load and process policy documents into vector store"""
    loader = DirectoryLoader(
        'docs/',
        glob="**/*.md",
        loader_cls=TextLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

class DatabaseTools:
    def __init__(self, db_path="loan_servicing.db"):
        print(f"Initializing DatabaseTools with path: {db_path}")  # Debug line
        self.db_path = db_path
        # Verify database connection on initialization
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Available tables: {tables}")  # Debug line
            conn.close()
        except Exception as e:
            print(f"Database initialization error: {str(e)}")
            raise
    
    def query_customer_info(self, customer_id: str) -> str:
        """Query customer information from database"""
        try:
            # Validate customer_id format
            customer_id = str(customer_id).strip()
            if not customer_id:
                return "Invalid customer ID provided"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
            SELECT 
                c.customer_id,
                c.name,
                c.contact_info,
                l.loan_amount,
                l.term,
                pb.remaining_balance
            FROM Customers c
            LEFT JOIN Loans l ON c.customer_id = l.customer_id
            LEFT JOIN PaymentBalances pb ON l.loan_id = pb.loan_id
            WHERE c.customer_id = ?
            """
            
            cursor.execute(query, (customer_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return f"Customer Info: ID={result[0]}, Name={result[1]}, Contact={result[2]}, Loan Amount=${result[3]:,.2f}, Term={result[4]} months, Remaining Balance=${result[5]:,.2f}"
            return "Customer not found"
        except Exception as e:
            return f"Error accessing customer information: {str(e)}"

    def query_payment_history(self, customer_id: str) -> str:
        """Query payment history for a customer"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            p.payment_id,
            p.amount,
            p.date
        FROM Payments p
        JOIN Loans l ON p.loan_id = l.loan_id
        WHERE l.customer_id = ?
        ORDER BY p.date DESC
        LIMIT 5
        """
        
        cursor.execute(query, (customer_id,))
        results = cursor.fetchall()
        conn.close()
        
        if results:
            history = "Recent Payment History:\n"
            for payment in results:
                history += f"Payment ID: {payment[0]}, Amount: ${payment[1]:,.2f}, Date: {payment[2]}\n"
            return history
        return "No payment history found"

    def query_payment_balances(self, customer_id: str) -> str:
        """Query detailed payment balances for a customer"""
        print(f"Querying payment balances for customer ID: {customer_id}")  # Debug line
        try:
            # Validate customer_id format
            customer_id = str(customer_id).strip()
            if not customer_id:
                return "Invalid customer ID provided"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # First verify customer exists
            cursor.execute("SELECT customer_id FROM Customers WHERE customer_id = ?", (customer_id,))
            if not cursor.fetchone():
                conn.close()
                return f"No customer found with ID {customer_id}"
            
            query = """
            SELECT 
                c.name,
                l.loan_id,
                pb.total_loan_amount,
                pb.total_payments_made,
                pb.remaining_balance,
                l.term,
                ROUND(CAST(pb.total_payments_made AS FLOAT) / pb.total_loan_amount * 100, 2) as payment_progress
            FROM Customers c
            JOIN Loans l ON c.customer_id = l.customer_id
            JOIN PaymentBalances pb ON l.loan_id = pb.loan_id
            WHERE c.customer_id = ?
            """
            
            cursor.execute(query, (customer_id,))
            results = cursor.fetchall()
            conn.close()
            
            if results:
                response = "Payment Balance Details:\n"
                for result in results:
                    response += f"Customer: {result[0]}\n"
                    response += f"Loan ID: {result[1]}\n"
                    response += f"Total Loan Amount: ${result[2]:,.2f}\n"
                    response += f"Total Payments Made: ${result[3]:,.2f}\n"
                    response += f"Remaining Balance: ${result[4]:,.2f}\n"
                    response += f"Loan Term: {result[5]} months\n"
                    response += f"Payment Progress: {result[6]}%\n\n"
                return response
            return "No payment balance information found for this customer"
            
        except sqlite3.Error as e:
            print(f"Database error: {str(e)}")  # Debug line
            return f"Error accessing payment balance information: {str(e)}"
        except Exception as e:
            print(f"General error: {str(e)}")  # Debug line
            return f"Error processing payment balance information: {str(e)}"

class ConversationState(TypedDict):
    messages: Annotated[List, add_messages]  # Chat history with add_messages reducer
    current_input: str  # Current user input
    query_type: Dict  # Classification of the query
    customer_data: Dict  # Results from customer data tools
    policy_data: Dict  # Results from policy search
    final_response: str  # Final response to user
    next_node: str | None  # Next node in the graph for routing

def create_classifier_node(llm):
    """Creates a node that classifies incoming queries"""
    def classify_query(state: ConversationState) -> ConversationState:
        # Get the latest message and conversation history
        latest_message = state["current_input"]
        messages = state.get("messages", [])
        
        # Check if the current input is a numeric value
        import re
        current_input_id = None
        if latest_message.strip().isdigit():
            current_input_id = latest_message.strip()
        
        # Check if we previously asked for a customer ID
        previously_requested_id = False
        if len(messages) >= 2:
            # Convert LangChain message objects to dict if needed
            last_assistant_msg = None
            for msg in reversed(messages):
                if isinstance(msg, (dict, HumanMessage, AIMessage)):
                    if hasattr(msg, 'content'):
                        content = msg.content
                        role = 'assistant' if isinstance(msg, AIMessage) else 'user'
                    else:
                        content = msg.get('content', '')
                        role = msg.get('role', '')
                    
                    if role == 'assistant':
                        last_assistant_msg = content
                        break
            
            if last_assistant_msg and any(phrase in last_assistant_msg.lower() 
                                        for phrase in ["need your customer id", "provide your customer id",
                                                     "share your customer id"]):
                previously_requested_id = True
        
        # If this is a numeric response to a request for ID, treat it as customer_data
        if previously_requested_id and current_input_id:
            classification = {
                "query_type": "customer_data",
                "customer_id": current_input_id,
                "policy_area": None
            }
            state["query_type"] = classification
            return state
        
        # Check conversation history for previously provided customer ID
        customer_id = None
        for msg in reversed(messages):
            # Handle both dict and Message objects
            if isinstance(msg, (dict, HumanMessage, AIMessage)):
                if hasattr(msg, 'content'):
                    content = msg.content
                    role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                else:
                    content = msg.get('content', '')
                    role = msg.get('role', '')
                
                if role == "user":
                    content = content.lower()
                    # Look for patterns like "customer id: 123" or "id: 123" or just "123"
                    if "customer id" in content or "id:" in content:
                        matches = re.findall(r'\d+', content)
                        if matches:
                            customer_id = matches[0]
                            break
                    # Also check for standalone numbers in previous messages
                    elif content.strip().isdigit():
                        customer_id = content.strip()
                        break
        
        # Create a zero-temperature LLM instance for classification
        classifier_llm = ChatGroq(
            temperature=0,
            model_name="gemma2-9b-it"
        )
        
        # Rest of the classification logic
        classification_prompt = """RESPOND ONLY WITH A PLAIN JSON OBJECT. NO MARKDOWN, NO EXPLANATION, NO CODE BLOCKS.

        Context: The user's question should be classified and include any customer ID found in the conversation history.
        Previous assistant response requested customer ID: {previously_requested_id}
        Current input is numeric: {is_numeric}
        
        Rules for classification:
        1. If previous message asked for ID and current input is numeric = customer_data with that ID
        2. If query is about account/loan/payment/balance AND we have a customer ID = customer_data
        3. If query is about account/loan/payment/balance with NO customer ID = customer_data (will prompt for ID)
        4. If query is about policies/regulations = policy
        5. If it's general chat/greetings = general
        
        Current customer ID from conversation: {customer_id}
        Current input as potential ID: {current_input_id}
        Query to classify: {query}

        Expected format (examples):
        {{"query_type":"customer_data","customer_id":"123","policy_area":null}}
        {{"query_type":"customer_data","customer_id":null,"policy_area":null}}
        {{"query_type":"policy","customer_id":null,"policy_area":"late_payment"}}
        {{"query_type":"general","customer_id":null,"policy_area":null}}"""
        
        try:
            # Get response content from LLM
            response = classifier_llm.invoke(
                classification_prompt.format(
                    query=latest_message,
                    customer_id=customer_id if customer_id else "None",
                    current_input_id=current_input_id if current_input_id else "None",
                    previously_requested_id=str(previously_requested_id),
                    is_numeric=str(bool(current_input_id))
                )
            )
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean up the response text
            json_content = response_text.strip()
            
            # Try to find JSON if there's any surrounding text
            if '{' in json_content:
                start = json_content.find('{')
                end = json_content.rfind('}') + 1
                json_content = json_content[start:end]
            
            # Parse the JSON
            classification = json.loads(json_content)
            
            # Override classification if we have a numeric response to an ID request
            if previously_requested_id and current_input_id:
                classification = {
                    "query_type": "customer_data",
                    "customer_id": current_input_id,
                    "policy_area": None
                }
            # Ensure customer_id from history is included if present
            elif customer_id and classification["query_type"] == "customer_data":
                classification["customer_id"] = customer_id
            
            # Validate required fields
            if "query_type" not in classification:
                raise ValueError("Missing query_type in classification")
                
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            print(f"Classification error: {str(e)}, Response: {response_text}")
            # Fallback classification
            classification = {
                "query_type": "general",
                "customer_id": current_input_id or customer_id,  # Include any available ID
                "policy_area": None
            }
        
        state["query_type"] = classification
        return state

    return classify_query

def create_router_node():
    """Creates a router node that directs the flow based on query classification"""
    def route_query(state: ConversationState) -> ConversationState:
        query_type = state["query_type"]["query_type"]
        customer_id = state["query_type"].get("customer_id")
        
        print(f"Router executing - Type: {query_type}, ID: {customer_id}")
        
        # Direct routing based on query type and customer ID
        if query_type == "customer_data" and customer_id:
            print(f"Router: Directing to customer_processor with ID {customer_id}")
            state["next_node"] = "customer_processor"
        elif query_type == "policy":
            state["next_node"] = "policy_processor"
        else:
            state["next_node"] = "response_compiler"
        
        return state
    return route_query

def create_customer_node(db_tools: DatabaseTools):
    """Creates a node that handles customer data retrieval"""
    def process_customer_query(state: ConversationState) -> ConversationState:
        print("Customer processor executing with state:", {
            "query_type": state["query_type"],
            "current_input": state["current_input"],
            "next_node": state.get("next_node")
        })  # Debug line
        
        if state["query_type"]["query_type"] != "customer_data":
            print("Skipping customer processor - not customer_data type")  # Debug line
            state["customer_data"] = {}
            return state
            
        customer_id = state["query_type"].get("customer_id")
        current_input = state["current_input"].lower()
        
        if not customer_id:
            print("No customer ID found in state")  # Debug line
            state["customer_data"] = {
                "error": "Customer ID required",
                "message": "I need your customer ID to look up that information. Could you please provide it?"
            }
            return state
            
        print(f"Processing customer ID: {customer_id}")  # Debug line
        
        # Always query payment balances when we have a customer ID
        payment_balances = db_tools.query_payment_balances(customer_id)
        print(f"Retrieved payment balances: {payment_balances}")  # Debug line
        
        state["customer_data"] = {
            "payment_balances": payment_balances
        }
        
        # If this is a new ID being provided, also get basic customer info
        if current_input.strip().isdigit():
            customer_info = db_tools.query_customer_info(customer_id)
            print(f"Retrieved customer info: {customer_info}")  # Debug line
            state["customer_data"]["customer_info"] = customer_info
            
        # Add payment history for specific queries
        if any(word in current_input for word in ["payment", "history", "paid"]):
            payment_history = db_tools.query_payment_history(customer_id)
            print(f"Retrieved payment history: {payment_history}")  # Debug line
            state["customer_data"]["payment_history"] = payment_history
            
        return state
    
    return process_customer_query

def create_policy_node(vectorstore: Chroma, llm):
    """Creates a node that handles policy-related queries"""
    def process_policy_query(state: ConversationState) -> ConversationState:
        if state["query_type"]["query_type"] != "policy":
            state["policy_data"] = {}
            # Preserve any existing final_response
            if "final_response" in state:
                state["final_response"] = state["final_response"]
            return state
            
        query = state["current_input"]
        docs = vectorstore.similarity_search(query)
        
        synthesis_prompt = """Based on these policy documents, answer the user's question:
        Question: {question}
        
        Relevant Policies:
        {docs}
        
        Provide a clear, concise answer that accurately reflects our policies."""
        
        response = llm.invoke(
            synthesis_prompt.format(
                question=query,
                docs="\n".join([doc.page_content for doc in docs])
            )
        )
        
        state["policy_data"] = {"response": response}
        return state
    
    return process_policy_query

def create_response_compiler(llm):
    """Creates a function to compile final response from various nodes"""
    def compile_response(state: ConversationState) -> ConversationState:
        try:
            # Check if this is a follow-up question about required information
            if state["current_input"].lower().strip() in ["what info do you want", "what information do you need", "what do you need"]:
                state["final_response"] = """To check your loan details or balance, I need your customer ID. 
                This is a unique number assigned to your account. 
                Once you provide your customer ID, I can tell you:
                - Your total outstanding balance
                - Payment history
                - Loan details
                
                Please share your customer ID with me."""
                return state

            # Regular response compilation logic
            if state["query_type"]["query_type"] == "customer_data":
                if "error" in state["customer_data"]:
                    state["final_response"] = state["customer_data"].get("message", 
                        "I need your customer ID to look up that information. Could you please provide it?")
                else:
                    response = ""
                    for key, value in state["customer_data"].items():
                        response += value + "\n\n"
                    state["final_response"] = response.strip()
                    
            elif state["query_type"]["query_type"] == "policy":
                policy_response = state["policy_data"].get("response")
                if policy_response:
                    state["final_response"] = policy_response.content if hasattr(policy_response, 'content') else str(policy_response)
                else:
                    state["final_response"] = "I couldn't find any relevant policy information. Could you please rephrase your question?"
                
            else:  # general query
                state["final_response"] = """I can help you with:
                - Looking up your loan details and current balance
                - Checking your payment history
                - Answering questions about our policies
                
                To access your account information, I'll need your customer ID. Please provide it, and I'll be happy to help."""
            
            # Ensure we have a valid response
            if not state["final_response"]:
                state["final_response"] = "I need more information to help you. Could you please provide more details?"
            
        except Exception as e:
            print(f"Error in response compiler: {str(e)}")
            state["final_response"] = "I encountered an error while processing your request. Could you please try again?"
            
        return state
    return compile_response

def create_graph(llm, vectorstore, db_tools):
    """Creates the main processing graph"""
    from langgraph.graph.message import add_messages
    
    # Configure graph with proper message handling for this LangGraph version
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("classifier", create_classifier_node(llm))
    workflow.add_node("router", create_router_node())
    workflow.add_node("customer_processor", create_customer_node(db_tools))
    workflow.add_node("policy_processor", create_policy_node(vectorstore, llm))
    workflow.add_node("response_compiler", create_response_compiler(llm))
    
    # Basic flow with direct paths
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", "router")
    #workflow.add_edge("router", "customer_processor")
    #workflow.add_edge("router", "policy_processor")
    #workflow.add_edge("router", "response_compiler")
    workflow.add_edge("customer_processor", "response_compiler")
    workflow.add_edge("policy_processor", "response_compiler")
    workflow.add_edge("response_compiler", END)
    
    # Define routing function that respects direct paths
    def route_by_type(state):
        query_type = state["query_type"]["query_type"]
        customer_id = state["query_type"].get("customer_id")
        print(f"\n=== Graph Routing Decision ===")
        print(f"Input: {state['current_input']}")
        print(f"Type: {query_type}")
        print(f"ID: {customer_id}")
        
        # Let direct edges handle customer data routing
        if query_type == "customer_data" and customer_id:
            print("Using direct edge to customer_processor")
            return "customer_processor"
        elif query_type == "policy":
            return "policy_processor"
            
        return "response_compiler"
    
    # Add conditional routing without overriding direct paths
    workflow.add_conditional_edges(
        "router",
        route_by_type
    )
    
    return workflow.compile()

def main():
    st.set_page_config(
        page_title="Loan Servicing Assistant",
        page_icon="💬",
        layout="wide"
    )
    
    # Add debug panel in sidebar
    with st.sidebar:
        st.title("Debug Panel")
        show_thoughts = st.checkbox("Show Agent Thought Process", value=False)
    
    st.title("Loan Servicing Assistant")
    st.markdown("Welcome to the Loan Servicing Assistant! I can help you with loan details, payment history, and policy questions.")

    # Initialize session state if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I am your loan servicing assistant. I can help you with loan details, payment history, and policy questions. How can I assist you today?"}
        ]
    
    if 'graph' not in st.session_state:
        # Initialize LLM
        llm = ChatGroq(
            temperature=0.2,
            model_name="gemma2-9b-it"
        )
        
        # Load policy documents into vector store
        vectorstore = load_policy_documents()
        
        # Initialize database tools
        db_tools = DatabaseTools()
        
        # Create the graph
        st.session_state.graph = create_graph(llm, vectorstore, db_tools)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your loan..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    
                    # Convert message history to proper format for LangGraph
                    formatted_messages = []
                    for msg in st.session_state.messages:
                        if isinstance(msg, (HumanMessage, AIMessage)):
                            formatted_messages.append({
                                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                                "content": msg.content
                            })
                        else:
                            formatted_messages.append(msg)
                    
                    # Initialize state with all required fields
                    initial_state: ConversationState = {
                        "messages": formatted_messages,
                        "current_input": prompt,
                        "query_type": {"query_type": "general", "customer_id": None, "policy_area": None},
                        "customer_data": {},
                        "policy_data": {},
                        "final_response": "",
                        "next_node": None
                    }
                    
                    # Process through graph
                    final_state = None
                    processed_states = []
                    
                    # Debug logger for state transitions
                    def log_state(step, phase):
                        print(f"\nState at {phase}:")
                        print(f"Node: {step.get('next_node')}")
                        print(f"Query type: {step.get('query_type')}")
                        print(f"Customer data: {step.get('customer_data')}")
                        print(f"Final response: {step.get('final_response')}")
                    
                    for step in st.session_state.graph.stream(initial_state):
                        log_state(step, "Current step")  # Debug logging
                        
                        if show_thoughts:
                            with st.sidebar:
                                st.markdown("### Processing Step")
                                st.markdown(f"Node: {step.get('next_node', 'unknown')}")
                                st.json({k: v for k, v in step.items() if k != "messages"})
                        
                        processed_states.append(step)
                        final_state = step
                    
                    if final_state and final_state.get("response_compiler"):
                        response = final_state["response_compiler"].get("final_response")
                        if response:
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.error("I apologize, but I couldn't find an answer to your question.")
                    
                    
                    
                    # Check timeout
                    if time.time() - start_time >= 3000:
                        st.error("I apologize, but the response took too long. Please try again or contact a human agent.")
                        st.info(f"Conversation ID: {hash(str(st.session_state.messages))}")
                        return
                    
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                st.error(f"I apologize, but I encountered an error: {str(e)}")
                st.button("Connect to Human Agent", on_click=lambda: st.info("Connecting to a human agent..."))

if __name__ == "__main__":
    main()
