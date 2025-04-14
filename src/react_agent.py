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

def main():
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
        )
    ]
    
    # Create memory with state tracking
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create agent prompt
    prompt = PromptTemplate.from_template(
        """You are a helpful loan servicing assistant with access to customer information and company policies.
        You can look up customer details, payment history, and reference company policies to answer questions.
        Always be professional and follow the company's policies when providing information.
        
        You have access to the following tools:
        {tools}
        
        When responding you MUST use this exact format:
        Thought: First analyze if:
        1. You need a tool
        2. You can answer directly
        3. You need more information
        - Before asking for information, check the chat history to see if it was already provided
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
    
    # Create agent with improved error handling
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create agent executor with memory and improved error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
        early_stopping_method="force",
        max_execution_time=30,  # 30 seconds timeout
        return_intermediate_steps=True
    )

    # Store initial context in memory
    memory.chat_memory.add_ai_message(
        "I am your loan servicing assistant. To access your account information, I'll need your customer ID. How can I help you today?"
    )
    
    return agent_executor, memory

if __name__ == "__main__":
    agent_executor, memory = main()
    
    # Interactive loop
    print("Loan Servicing Assistant Ready! (Type 'quit' to exit)")
    while True:
        user_input = input("Question: ")
        if user_input.lower() == 'quit':
            break
        
        try:
            start_time = time.time()
            response = agent_executor.invoke({"input": user_input})
            print("\nAssistant:", response["output"])
            
            # Check if max iterations were reached
            if response.get("intermediate_steps") and len(response["intermediate_steps"]) >= agent_executor.max_iterations:
                raise IterationLimitError("Maximum number of iterations reached")
            
            # Check if we exceeded the time limit
            if time.time() - start_time >= agent_executor.max_execution_time:
                raise TimeoutError("Response time exceeded limit")
                
        except (IterationLimitError, TimeoutError) as e:
            print("\nI apologize, but I need to transfer you to a human agent for better assistance.")
            print("A customer service representative will be with you shortly.")
            print("For reference, your conversation has been saved with ID: ", hash(str(memory.chat_memory)))
            break
        except Exception as e:
            print(f"I apologize, but I encountered an error: {str(e)}")
            print("Would you like to try again or speak with a human agent? (Type 'human' for human agent, or continue asking questions)")
            if input().lower() == 'human':
                print("\nTransferring you to a human agent...")
                print("A customer service representative will be with you shortly.")
                print("For reference, your conversation has been saved with ID: ", hash(str(memory.chat_memory)))
                break
