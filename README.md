# Hackathon-EDW
Code file for EDW hackathon

# Problem Statement:
Predict the Foreclosure of a loan.

Foreclosure:
Given the availability of various alternatives across the industry customer has a propensity to move to another financial institution for Balance Transfer. Foreclosure and balance transfer has added to the concerns of NBFCs. Foreclosure means repaying the outstanding loan amount in a single payment instead of with EMIs while balance transfer is transferring outstanding Loan availed from one Bank / Financial Institution to another Bank / Financial Institution, usually on the grounds of better service, top-up on the existing loan, proximity of branch, saving on interest repayments, etc. Losing out on customers on grounds on foreclosure and balance transfer leads to revenue loss. Acquiring a new customer can cost up to five times more than retaining an existing customer and an increase in customer retention by 5% increases profits up to 25%.



# Solution
Available data 
1. Customer demographic data as Cust
2. LMS ( transaction data)
3. RF( Email_data)
4. Train
5. Test

# Starting with Customer demographic data and LMS data


Glimpse of data and its uni-variate stats is in the file("EDA on cust and LMS data")

Dimension(LMS)
38  Variables      624250  Observations

Dimension (Cust)
13  Variables      10000  Observations

Common variable 
Cust_id

13 var in cust carries customer based information so we have to join the tables LMS and Cust.
Instead i have segment the customer table and use the cluster only as a new variable in LMS data, and then will see the correlation with
target Var.





# Key points after EDA
1. Train data is skewed with only 9% event has happened
2. Cust data is insufficent to map all the customer present in LMS data.
3. So I clustered them and idea is to replacing missing value of one col is precise as well as less time consuming as compared to a table.

Further details will be updated soon, Codes are linked.

