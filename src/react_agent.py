import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
import sqlite3

load_dotenv()

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
    
    # Create agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create agent executor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,  # Handle parsing errors gracefully
        max_iterations=5  # Prevent infinite loops
    )

    # Store initial context in memory
    memory.chat_memory.add_ai_message("I am your loan servicing assistant. How can I help you today?")
    
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
            response = agent_executor.invoke({"input": user_input})
            print("\nAssistant:", response["output"])
        except Exception as e:
            print(f"Error: {str(e)}")
