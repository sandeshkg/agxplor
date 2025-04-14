# Loan Servicing Assistant

An AI-powered loan servicing assistant that helps customers with loan inquiries, payment history, and policy questions.

## Features

- Interactive chat interface using Streamlit
- Real-time access to customer loan information
- Policy search capabilities using vector embeddings
- Detailed payment history and balance tracking
- Debug panel for viewing the assistant's thought process

## Prerequisites

- Python 3.8 or higher
- Poetry (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agxplor
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Database Setup

1. Initialize the SQLite database with sample data:
```bash
poetry run python db/populate_database.py
```

This will create `loan_servicing.db` with tables for:
- Customers
- Loans
- Payments
- Payment Balances
- Interest Rates
- And more...

## Running the Application

1. Start the Streamlit interface:
```bash
poetry run streamlit run src/streamlit_react_agent.py
```

The application will:
- Load policy documents from the `docs/` directory into a vector store (automatically created at runtime)
- Create a chat interface with a debug panel in the sidebar
- Connect to the SQLite database for customer information

## Development and Debugging

1. Enable debug mode to see the assistant's thought process:
- Use the checkbox in the sidebar labeled "Show Agent Thought Process"
- This will display each step of reasoning, tools used, and observations

2. Run the application in debug mode:
```bash
poetry run python -m debugpy --listen 5678 src/streamlit_react_agent.py
```

3. View logs in development:
```bash
poetry run streamlit run src/streamlit_react_agent.py --logger.level=debug
```

## Environment Variables

Create a `.env` file with:
```
GROQ_API_KEY=your_api_key_here
```

## Project Structure

- `src/`: Source code files
  - `streamlit_react_agent.py`: Main Streamlit application
  - `react_agent.py`: Core agent logic
- `docs/`: Policy documents in Markdown format
- `db/`: Database schema and population scripts
- `chroma_db/`: Vector store (automatically created at runtime)

## Testing

Run tests using Poetry:
```bash
poetry run pytest
```