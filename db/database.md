# Loan Servicing for Vehicles - Database Tables

| Table Name           | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Customers            | Stores customer details such as name, contact information, and address.    |
| Loans                | Contains loan details including loan ID, customer ID, loan amount, and term.|
| Vehicles             | Stores information about vehicles such as make, model, year, and VIN.      |
| Payments             | Tracks loan payments including payment ID, loan ID, amount, and date.      |
| Delinquencies        | Records overdue payments and delinquency status for loans.                 |
| Collaterals          | Details about collateral vehicles tied to loans.                           |
| InterestRates        | Stores interest rate details based on loan type and term.                  |
| LoanApplications     | Tracks loan application details and their approval status.                 |
| InsurancePolicies    | Links insurance details to vehicles and loans.                             |
| Repossessions        | Records details of repossessed vehicles due to loan default.               |

## Table Columns

### Customers
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| customer_id      | INT         | Unique identifier for the customer.      |
| name             | VARCHAR(255)| Full name of the customer.               |
| contact_info     | TEXT        | Contact details such as phone and email. |
| address          | TEXT        | Residential address of the customer.     |

### Loans
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| loan_id          | INT         | Unique identifier for the loan.          |
| customer_id      | INT         | Identifier linking to the customer.      |
| loan_amount      | DECIMAL(10,2)| Total amount of the loan.                |
| term             | INT         | Loan term in months.                     |

### Vehicles
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| vehicle_id       | INT         | Unique identifier for the vehicle.       |
| make             | VARCHAR(255)| Manufacturer of the vehicle.             |
| model            | VARCHAR(255)| Model of the vehicle.                    |
| year             | INT         | Manufacturing year of the vehicle.       |
| vin              | VARCHAR(17) | Vehicle Identification Number.           |

### Payments
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| payment_id       | INT         | Unique identifier for the payment.       |
| loan_id          | INT         | Identifier linking to the loan.          |
| amount           | DECIMAL(10,2)| Payment amount.                          |
| date             | DATE        | Date of the payment.                     |

### Delinquencies
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| delinquency_id   | INT         | Unique identifier for the delinquency.   |
| loan_id          | INT         | Identifier linking to the loan.          |
| overdue_amount   | DECIMAL(10,2)| Amount overdue.                          |
| status           | VARCHAR(50) | Status of the delinquency.               |

### Collaterals
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| collateral_id    | INT         | Unique identifier for the collateral.    |
| loan_id          | INT         | Identifier linking to the loan.          |
| vehicle_id       | INT         | Identifier linking to the vehicle.       |

### InterestRates
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| rate_id          | INT         | Unique identifier for the interest rate. |
| loan_type        | VARCHAR(50) | Type of loan.                            |
| rate             | DECIMAL(5,2)| Interest rate percentage.                |
| term             | INT         | Loan term in months.                     |

### LoanApplications
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| application_id   | INT         | Unique identifier for the application.   |
| customer_id      | INT         | Identifier linking to the customer.      |
| loan_amount      | DECIMAL(10,2)| Requested loan amount.                   |
| status           | VARCHAR(50) | Approval status of the application.      |

### InsurancePolicies
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| policy_id        | INT         | Unique identifier for the policy.        |
| vehicle_id       | INT         | Identifier linking to the vehicle.       |
| loan_id          | INT         | Identifier linking to the loan.          |
| policy_details   | TEXT        | Details of the insurance policy.         |

### Repossessions
| Column Name      | Data Type   | Description                              |
|------------------|-------------|------------------------------------------|
| repossession_id  | INT         | Unique identifier for the repossession.  |
| loan_id          | INT         | Identifier linking to the loan.          |
| vehicle_id       | INT         | Identifier linking to the vehicle.       |
| date             | DATE        | Date of the repossession.                |