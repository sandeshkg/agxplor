import sqlite3
import random
from datetime import datetime, timedelta
import faker

# Initialize Faker instance
fake = faker.Faker()

# Connect to SQLite database
conn = sqlite3.connect('loan_servicing.db')
cursor = conn.cursor()

# Drop existing tables if they exist
cursor.executescript('''
DROP TABLE IF EXISTS InsurancePolicies;
DROP TABLE IF EXISTS Repossessions;
DROP TABLE IF EXISTS PaymentBalances;
DROP TABLE IF EXISTS Delinquencies;
DROP TABLE IF EXISTS Payments;
DROP TABLE IF EXISTS Collaterals;
DROP TABLE IF EXISTS LoanApplications;
DROP TABLE IF EXISTS Loans;
DROP TABLE IF EXISTS InterestRates;
DROP TABLE IF EXISTS Vehicles;
DROP TABLE IF EXISTS Customers;
''')

# Create tables
cursor.execute('''
CREATE TABLE Customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    contact_info TEXT,
    address TEXT
);
''')

cursor.execute('''
CREATE TABLE Loans (
    loan_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    loan_amount REAL,
    term INTEGER,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
''')

cursor.execute('''
CREATE TABLE Vehicles (
    vehicle_id INTEGER PRIMARY KEY,
    make TEXT,
    model TEXT,
    year INTEGER,
    vin TEXT
);
''')

cursor.execute('''
CREATE TABLE Payments (
    payment_id INTEGER PRIMARY KEY,
    loan_id INTEGER,
    amount REAL,
    date TEXT,
    FOREIGN KEY (loan_id) REFERENCES Loans(loan_id)
);
''')

cursor.execute('''
CREATE TABLE Delinquencies (
    delinquency_id INTEGER PRIMARY KEY,
    loan_id INTEGER,
    overdue_amount REAL,
    status TEXT,
    FOREIGN KEY (loan_id) REFERENCES Loans(loan_id)
);
''')

cursor.execute('''
CREATE TABLE Collaterals (
    collateral_id INTEGER PRIMARY KEY,
    loan_id INTEGER,
    vehicle_id INTEGER,
    FOREIGN KEY (loan_id) REFERENCES Loans(loan_id),
    FOREIGN KEY (vehicle_id) REFERENCES Vehicles(vehicle_id)
);
''')

cursor.execute('''
CREATE TABLE InterestRates (
    rate_id INTEGER PRIMARY KEY,
    loan_type TEXT,
    rate REAL,
    term INTEGER
);
''')

cursor.execute('''
CREATE TABLE LoanApplications (
    application_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    loan_amount REAL,
    status TEXT,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
''')

cursor.execute('''
CREATE TABLE InsurancePolicies (
    policy_id INTEGER PRIMARY KEY,
    vehicle_id INTEGER,
    loan_id INTEGER,
    policy_details TEXT,
    FOREIGN KEY (vehicle_id) REFERENCES Vehicles(vehicle_id),
    FOREIGN KEY (loan_id) REFERENCES Loans(loan_id)
);
''')

cursor.execute('''
CREATE TABLE Repossessions (
    repossession_id INTEGER PRIMARY KEY,
    loan_id INTEGER,
    vehicle_id INTEGER,
    date TEXT,
    FOREIGN KEY (loan_id) REFERENCES Loans(loan_id),
    FOREIGN KEY (vehicle_id) REFERENCES Vehicles(vehicle_id)
);
''')

cursor.execute('''
CREATE TABLE PaymentBalances (
    balance_id INTEGER PRIMARY KEY,
    loan_id INTEGER,
    total_loan_amount REAL,
    total_payments_made REAL,
    remaining_balance REAL,
    FOREIGN KEY (loan_id) REFERENCES Loans(loan_id)
);
''')

# Generate Interest Rates
interest_rates = [
    (1, 'New Vehicle', 4.5, 36),
    (2, 'New Vehicle', 5.0, 48),
    (3, 'New Vehicle', 5.5, 60),
    (4, 'Used Vehicle', 5.5, 36),
    (5, 'Used Vehicle', 6.0, 48),
    (6, 'Used Vehicle', 6.5, 60),
]

# Generate data for 500 customers
customers = []
loans = []
vehicles = []
payments = []
delinquencies = []
collaterals = []
repossessions = []
loan_applications = []
insurance_policies = []

for i in range(1, 501):
    # Customer data
    name = fake.name()
    contact_info = fake.email()
    address = fake.address()
    customers.append((i, name, contact_info, address))

    # Vehicle data
    make = random.choice(["Toyota", "Ford", "Honda", "Chevrolet", "Nissan"])
    model = random.choice(["Camry", "F-150", "Civic", "Silverado", "Altima"])
    year = random.randint(2018, 2025)
    vin = fake.uuid4()[:17].upper()
    vehicles.append((i, make, model, year, vin))

    # Loan Application data
    loan_amount = round(random.uniform(5000, 50000), 2)
    application_status = random.choice(['Approved', 'Approved', 'Approved', 'Rejected', 'Pending'])
    loan_applications.append((i, i, loan_amount, application_status))

    # Create loan only if application was approved
    if application_status == 'Approved':
        term = random.choice([36, 48, 60])
        loans.append((i, i, loan_amount, term))
        
        # Create collateral entry
        collaterals.append((i, i, i))
        
        # Create insurance policy
        policy_types = ['Basic', 'Standard', 'Premium']
        coverage_amounts = [50000, 75000, 100000]
        policy_type = random.choice(policy_types)
        coverage = random.choice(coverage_amounts)
        policy_details = f"{policy_type} coverage - ${coverage:,}"
        insurance_policies.append((i, i, i, policy_details))

        # Generate payments
        payment_amount = round(loan_amount / term, 2)
        for j in range(random.randint(1, term)):
            payment_date = datetime.now() - timedelta(days=j * 30)
            payments.append((len(payments) + 1, i, payment_amount, payment_date.strftime('%Y-%m-%d')))

# Insert data into tables
cursor.executemany('INSERT INTO Customers VALUES (?, ?, ?, ?)', customers)
cursor.executemany('INSERT INTO Vehicles VALUES (?, ?, ?, ?, ?)', vehicles)
cursor.executemany('INSERT INTO LoanApplications VALUES (?, ?, ?, ?)', loan_applications)
cursor.executemany('INSERT INTO Loans VALUES (?, ?, ?, ?)', loans)
cursor.executemany('INSERT INTO Collaterals VALUES (?, ?, ?)', collaterals)
cursor.executemany('INSERT INTO InsurancePolicies VALUES (?, ?, ?, ?)', insurance_policies)
cursor.executemany('INSERT INTO Payments VALUES (?, ?, ?, ?)', payments)

# Generate delinquencies and repossessions
for loan in loans:
    if random.random() < 0.03:  # 3% delinquent within 30 days
        delinquency_status = "Overdue within 30 days"
        overdue_amount = round(random.uniform(100, 1000), 2)
        delinquencies.append((len(delinquencies) + 1, loan[0], overdue_amount, delinquency_status))
    elif random.random() < 0.03:  # 3% overdue more than 90 days
        delinquency_status = "Overdue more than 90 days"
        overdue_amount = round(random.uniform(100, 1000), 2)
        delinquencies.append((len(delinquencies) + 1, loan[0], overdue_amount, delinquency_status))
    if random.random() < 0.03:  # 3% repossessed vehicles
        repossession_date = datetime.now().strftime('%Y-%m-%d')
        repossessions.append((len(repossessions) + 1, loan[0], loan[0], repossession_date))

cursor.executemany('INSERT INTO Delinquencies VALUES (?, ?, ?, ?)', delinquencies)
cursor.executemany('INSERT INTO Repossessions VALUES (?, ?, ?, ?)', repossessions)

# Populate PaymentBalances table
payment_balances = []
for loan in loans:
    total_loan_amount = loan[2]  # loan_amount
    total_payments_made = sum(payment[2] for payment in payments if payment[1] == loan[0])
    remaining_balance = total_loan_amount - total_payments_made
    payment_balances.append((len(payment_balances) + 1, loan[0], total_loan_amount, total_payments_made, remaining_balance))

cursor.executemany('INSERT INTO PaymentBalances VALUES (?, ?, ?, ?, ?)', payment_balances)

# Insert new data
cursor.executemany('INSERT INTO InterestRates VALUES (?, ?, ?, ?)', interest_rates)

# Commit and close
conn.commit()
conn.close()