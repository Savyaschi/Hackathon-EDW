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
2. EDA on Cust data and LMS data reveals the pattern in LMS and Cust data, In Other words the corelation between Customer's Income and 
   Loan Taken is significant. And Similarly Customer's Age and Qualification does have similar corelation with Loan Amount.
   Takeaway from this EDA is Customer Profiling for Loan is going to add other cols in LMS data.
   Challenge 1: Cust Table has only 10000 cust and LMS has 33K unique Cust ID so Mapping them with Cust ID is going 23K* No of New          Variables will be added in LMS data.
   Aprroach: Since there is Corelation between these two tables so used Cust segmentation and created their Profile, there were 10          distinct cust type is found with Silhoute Coeff is .47, instead of merging entire cust table merged Customer cluster information to      LMS data. Using Random Forest/ KNN  /Missranger, missing values were imputed.
   
3. 


