import streamlit as st
import os
from typing import List
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
import sqlite3
import time

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
        self.db_path = db_path
    
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
        try:
            # Validate customer_id format
            customer_id = str(customer_id).strip()
            if not customer_id:
                return "Invalid customer ID provided"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
        except Exception as e:
            return f"Error accessing payment balance information: {str(e)}"

def main():
    st.set_page_config(
        page_title="Loan Servicing Assistant",
        page_icon="💬",
        layout="wide"  # Allow space for sidebar
    )
    
    # Add debug panel in sidebar
    with st.sidebar:
        st.title("Debug Panel")
        show_thoughts = st.checkbox("Show Agent Thought Process", value=False)
    
    st.title("Loan Servicing Assistant")
    st.markdown("Welcome to the Loan Servicing Assistant! I can help you with loan details, payment history, and policy questions.")

    # Initialize session state for agent and memory if not exists
    if 'agent_executor' not in st.session_state:
        # Initialize LLM
        llm = ChatGroq(
            temperature=0.2,
            model_name="gemma2-9b-it"
        )
        
        # Load policy documents into vector store
        vectorstore = load_policy_documents()
        
        # Initialize database tools
        db_tools = DatabaseTools()
        
        # Create tools list
        tools = [
            Tool(
                name="SearchPolicies",
                func=vectorstore.similarity_search,
                description="Useful for when you need to answer questions about company policies, including underwriting, fraud prevention, or compliance policies."
            ),
            Tool(
                name="CustomerInfo",
                func=db_tools.query_customer_info,
                description="Useful for getting customer information including loan details. Input should be a customer ID."
            ),
            Tool(
                name="PaymentHistory",
                func=db_tools.query_payment_history,
                description="Useful for getting a customer's payment history. Input should be a customer ID."
            ),
            Tool(
                name="PaymentBalances",
                func=db_tools.query_payment_balances,
                description="Useful for getting detailed payment balance information including total loan amount, payments made, remaining balance, and payment progress. Input should be a customer ID."
            )
        ]
        
        # Create memory with state tracking
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create agent with same prompt as before
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=PromptTemplate.from_template(
                """You are a helpful loan servicing assistant with access to customer information and company policies.
                You can look up customer details, payment history, and reference company policies to answer questions.
                Always be professional and follow the company's policies when providing information.
                
                You have access to the following tools:
                {tools}
                
                When responding you MUST use this exact format:
                Thought: First analyze if:
                1. You can answer directly
                2. You need a tool
                3. You need more information
                - Before asking for information, check the chat history to see if it was already provided
                - When you need multiple pieces of information, check the chat history to see if you can use any of it and only ask the missing information
                - If the information exists in chat history, use it instead of asking again
                - For vague questions, provide examples of information you can help with
                
                Action: If you need a tool, specify which tool to use from [{tool_names}]
                Action Input: The input for the tool
                Observation: The result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times if needed)
                Thought: I can now answer the human
                Final Answer: Your final response to the human

                If you need specific information like a customer ID AND it's not in chat history:
                Thought: I need more information to help and it's not in the chat history
                Final Answer: To help you with that, I'll need your customer ID. Could you please provide it?

                For vague questions like "where can I find" or general inquiries:
                Thought: Check the previous message to understand the context better and answer. If the question is vague or unrelated, I should provide helpful context about available information
                Final Answer: I can help you find various types of information, including:
                - Your loan details and current balance
                - Payment history
                - Account information
                Please let me know specifically what you're looking for, and I'll be happy to assist. If needed, I'll just need your customer ID to look up your information.

                Remember to:
                1. Always check chat history for any required information before asking the user
                2. If information was previously provided, use it directly
                3. Only ask for information that hasn't been shared before
                4. Keep track of context across the entire conversation
                5. Be proactive in explaining available information for vague questions

                Chat History: {chat_history}
                
                Human: {input}
                Assistant: Let me help you with that.
                {agent_scratchpad}"""
            )
        )
        
        # Create agent executor with memory and improved error handling
        st.session_state.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=st.session_state.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            early_stopping_method="force",
            max_execution_time=30,
            return_intermediate_steps=True
        )

        # Store initial context in memory
        st.session_state.memory.chat_memory.add_ai_message(
            "I am your loan servicing assistant. To access your account information, I'll need your customer ID. How can I help you today?"
        )
        
        # Initialize chat history
        st.session_state.messages = [
            {"role": "assistant", "content": "I am your loan servicing assistant. To access your account information, I'll need your customer ID. How can I help you today?"}
        ]

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
                    response = st.session_state.agent_executor.invoke({"input": prompt})
                    
                    # Show thought process in sidebar if enabled
                    if show_thoughts:
                        with st.sidebar:
                            st.markdown("### Current Thought Process")
                            
                            # Show full response
                            if "output" in response:
                                with st.expander("Final Response", expanded=True):
                                    st.markdown(f"**Final Answer:** {response['output']}")
                            
                            # Show all intermediate steps
                            if "intermediate_steps" in response:
                                for step in response["intermediate_steps"]:
                                    with st.expander(f"Step {response['intermediate_steps'].index(step) + 1}", expanded=True):
                                        # Show the complete thought process
                                        full_log = step[0].log
                                        if "\nAction:" in full_log:
                                            thought, rest = full_log.split("\nAction:", 1)
                                            st.markdown(f"**Initial Thought:** {thought}")
                                            st.markdown(f"**Action:** Action:{rest}")
                                        else:
                                            st.markdown(f"**Thought:** {full_log}")
                                        
                                        st.markdown(f"**Tool Used:** {step[0].tool}")
                                        st.markdown(f"**Tool Input:** {step[0].tool_input}")
                                        st.markdown(f"**Observation:** {step[1]}")
                    
                    # Check for errors
                    if response.get("intermediate_steps") and len(response["intermediate_steps"]) >= st.session_state.agent_executor.max_iterations:
                        st.error("I apologize, but I need to transfer you to a human agent for better assistance. A customer service representative will be with you shortly.")
                        st.info(f"Conversation ID: {hash(str(st.session_state.memory.chat_memory))}")
                        return
                    
                    if time.time() - start_time >= st.session_state.agent_executor.max_execution_time:
                        st.error("I apologize, but the response took too long. Please try again or contact a human agent.")
                        st.info(f"Conversation ID: {hash(str(st.session_state.memory.chat_memory))}")
                        return
                    
                    st.write(response["output"])
                    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                    
            except Exception as e:
                st.error(f"I apologize, but I encountered an error: {str(e)}")
                st.button("Connect to Human Agent", on_click=lambda: st.info("Connecting to a human agent..."))

if __name__ == "__main__":
    main()
