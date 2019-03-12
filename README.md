# Hackathon-EDW
Code file for EDW hackathon

# Problem Statement:
India is a diversified financial sector undergoing rapid expansion, both in terms of strong growth of existing financial services firms and new entities entering the market. The sector comprises commercial banks, insurance companies, non-banking financial companies, co-operatives, pension funds, mutual funds and other smaller financial entities .
The Edelweiss Group is one of India&#39;s leading diversified financial services company providing a broad range of financial products and services to a substantial and diversified client base that includes corporations, institutions and individuals. Edelweiss&#39;s products and services span multiple asset classes and consumer segments across domestic and global geographies.
Given the availability of various alternatives across the industry customer has a propensity to move to another financial institution for Balance Transfer. Foreclosure and balance transfer has added to the concerns of NBFCs. Foreclosure means repaying the outstanding loan amount in a single payment instead of with EMIs while balance transfer is transferring outstanding Loan availed from one Bank / Financial Institution to another Bank / Financial Institution, usually on the grounds of better service, top-up on the existing loan, proximity of branch, saving on interest repayments, etc. Losing out on customers on grounds on foreclosure and balance transfer leads to revenue loss. Acquiring a new customer can cost up to five times more than retaining an existing customer and an increase in customer retention by 5% increases profits up to 25%.
NBFCs have started taking pro-active measures to ensure this is curbed; and this is where you come in! Objective is primarily to arrive at a propensity to foreclose and balance transfer an existing loan based on lead indicators such as demographics, internal behavior and performance on all credit lines; along with the estimated ‘Time to Foreclose’. May the best algorithm win!


# Solution
Data Exploration:
1.Customer demographic data
2.LMS ( transaction data)
3.RF( Email_data)
4.Train
5.Test

# started with all the tables except Email data

glimpse of data and its univaariate stats is in the file("EDA on cust and LMS data")


# Key points after EDA
1. Train data is skewed with only 9% event has happened
2. Cust data is insufficent to map all the customer present in LMS data.
3. So I clustered them and idea is to replacing missing value of one col is precise as well as less time consuming as compared to a table.

Further details will be updated soon, Codes are linked.

