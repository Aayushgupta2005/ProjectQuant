import pandas as pd
import queue
from datetime import datetime
from collections import defaultdict
import numpy as np


df = pd.read_csv('RELIANCE.csv')
date = df['Date'].tolist()
opening_amt = df['Open'].tolist()
closing_amt=df['Close'].tolist()
high=df['High'].tolist()
low=df['Low'].tolist()
trades=[]
returns=[] 
returns_per_day=[]
num_profitable_trades=0
num_loss_trades=0

candles_for_profit=[]
candles_for_loss=[]

months_wise_gain_or_loss={
    "01":0,
    "02":0,
    "03":0,
    "04":0,
    "05":0,
    "06":0,
    "07":0,
    "08":0,
    "09":0,
    "10":0,
    "11":0,
    "12":0
    }
candles_req=[]
def risk_account_size_update(current_account_size,initial_account_size):
    price_after_20_percent_increment=(initial_account_size +(initial_account_size*0.2))
    price_after_20_percent_decrement=(initial_account_size -(initial_account_size*0.2))
    if(current_account_size>= price_after_20_percent_increment):
        return price_after_20_percent_increment
    elif(current_account_size<= price_after_20_percent_decrement):
        return price_after_20_percent_decrement
    else:
        return initial_account_size
    
purchase_history=[]
sell_history=[]
portfolio_value_history=[]
curr_trades=queue.PriorityQueue()
input_account=float(input("Enter yout initial amount : "))
main_account=input_account
risk_account=main_account
gain_array=[]
loss_array=[]
trading_durations=[]

def days_between(date1: str, date2: str) -> int:
    d1 = datetime.strptime(date1, "%Y-%m-%d")
    d2 = datetime.strptime(date2, "%Y-%m-%d")
    delta = d2 - d1
    return abs(delta.days)

nifty_data = pd.read_csv('nifty50_monthly_returns.csv')
monthly_dates_of_nifty= nifty_data['Date'].tolist()
nifty_50_returns= nifty_data['Return'].tolist()
monthly_returns_of_nifty_50=[]
for f in range(0,len(monthly_dates_of_nifty)):
    monthly_returns_of_nifty_50.append([monthly_dates_of_nifty[f],nifty_50_returns[f]])


def format_monthly_returns(data):
    formatted_returns = []
    for date_str, return_value in data:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        year_month = f"{date.year}-{date.month:02d}"
        formatted_return = f"{return_value:.2f}%"
        formatted_returns.append([year_month, formatted_return])
    return formatted_returns

monthly_returns_of_nifty_50_in_correct_format=format_monthly_returns(monthly_returns_of_nifty_50)
current_purchased_quantity=0
portfolio_value_including_purchased_stocks=[]
monthly_portfolio_values=[]

def calculate_monthly_returns(data):
        # Organize data by month
    monthly_data = defaultdict(list)
    for date_str, price in data:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        month_key = (date.year, date.month)
        monthly_data[month_key].append((date, price))
    
    # Calculate monthly returns
    monthly_returns = []
    for month_key, prices in monthly_data.items():
        prices.sort(key=lambda x: x[0])  # Sort by date
        start_price = prices[0][1]
        end_price = prices[-1][1]
        monthly_return = (end_price - start_price) / start_price
        monthly_returns.append([month_key[0], month_key[1], monthly_return])
    
    # Sort the returns by date
    monthly_returns.sort(key=lambda x: (x[0], x[1]))
    
    # Format the results
    formatted_returns = []
    for year, month, ret in monthly_returns:
        month_str = f'{year:04d}-{month:02d}'
        ret_str = f'{ret * 100:.2f}%'
        formatted_returns.append([month_str, ret_str])
    
    return formatted_returns



for i in range(0,len(date)-1):

    # let us first sell the trades which are hit either by stoploss or stopprofit....
    while(not(curr_trades.empty())):
        # print("current : ",current_purchased_quantity)
        # (stoploss_amount,[purchased_price,quantity_to_purchase,stoploss_amount,stopprofit_amount,date_purchased,i])   format of the element inside the queue
        temp_stop_loss_amount,temp_trade=curr_trades.get()
        temp_stop_loss_price=temp_trade[0]-temp_stop_loss_amount
        temp_stop_profit_price=temp_trade[0]+(3*temp_stop_loss_amount)
        temp_stop_profit_amount=3*temp_stop_loss_amount
        if(closing_amt[i]<=temp_stop_loss_price):
            #stoploss hit hence sell
            # profit_on_one_quantity=closing_amt[i]-purchased_price
            # total_profit=(quantity*profit_on_one_quantity) + total_profit
            date_sell=date[i]
            net_loss=temp_stop_loss_amount*temp_trade[1]
            amount_credit_to_bank=(temp_trade[0]-temp_stop_loss_amount)*temp_trade[1]
            main_account=main_account+(amount_credit_to_bank)
            portfolio_value_history.append(main_account)
            sell_history.append(amount_credit_to_bank)
            date_purchased=temp_trade[4]
            risk_account=risk_account_size_update(main_account,risk_account)
            days_taken=days_between(date_purchased,date_sell)
            trading_durations.append(days_taken)
            trades.append([date_purchased,date_sell,"quantity : ",temp_trade[1],"loss",net_loss,main_account,((i-temp_trade[-1])+1),"days taken : ",days_taken])
            returns.append(-(net_loss/(temp_trade[0]*temp_trade[1]))*100)
            returns_per_day.append((-(net_loss/(temp_trade[0]*temp_trade[1]))*100)/days_taken)
            loss_array.append(net_loss)
            num_loss_trades=num_loss_trades+1
            candles_for_loss.append((i-temp_trade[-1])+1)
            candles_req.append((i-temp_trade[-1])+1)
            months_wise_gain_or_loss[date[i][-5:-3]]-=net_loss
            current_purchased_quantity=current_purchased_quantity-temp_trade[1]
            # initial_amt=initial_amt+(quantity*profit_on_one_quantity)
            # trades.append([date_purchased,date_sell,quantity*profit_on_one_quantity,initial_amt])

            
        elif(closing_amt[i]>=temp_stop_profit_price):
            #stopprofit hit hence sell
            # profit_on_one_quantity=closing_amt[i]-purchased_price
            # total_profit=(quantity*profit_on_one_quantity) + total_profit
            date_sell=date[i]
            # print(temp_trade[1])
            net_gain=temp_stop_profit_amount*temp_trade[1]
            # print(net_gain)
            amount_credit_to_bank=(temp_stop_profit_price)*temp_trade[1]
            main_account=main_account+(amount_credit_to_bank)
            portfolio_value_history.append(main_account)
            sell_history.append(amount_credit_to_bank)
            date_purchased=temp_trade[4]
            risk_account=risk_account_size_update(main_account,risk_account)
            days_taken=days_between(date_purchased,date_sell)
            trading_durations.append(days_taken)
            trades.append([date_purchased,date_sell,"quantity : ",temp_trade[1],"gain",net_gain,main_account,((i-temp_trade[-1])+1),"days taken : ",days_taken])
            returns.append((net_gain/(temp_trade[0]*temp_trade[1]))*100)
            returns_per_day.append(((net_gain/(temp_trade[0]*temp_trade[1]))*100)/days_taken)
            gain_array.append(net_gain)
            num_profitable_trades=num_profitable_trades+1
            candles_for_profit.append((i-temp_trade[-1])+1)
            months_wise_gain_or_loss[date[i][-5:-3]]+=net_gain
            candles_req.append((i-temp_trade[-1])+1)
            current_purchased_quantity=current_purchased_quantity-temp_trade[1]
            # initial_amt=initial_amt+(quantity*profit_on_one_quantity)
            # trades.append([date_purchased,date_sell,quantity*profit_on_one_quantity,initial_amt])
        else:
            curr_trades.put((temp_stop_loss_amount,temp_trade))
            break

    portfolio_value_including_purchased_stocks.append([date[i],main_account+(closing_amt[i]*current_purchased_quantity)])
    # [date[i],main_account+(closing_amt[i]*current_purchased_quantity)]
    
    #Checking if there is any possibility to trade and if yes then make a purchase
    if((closing_amt[i] - opening_amt[i]) < 0) :   #sorted !!
        #that means the candle is red.
        #check if the adjascent candle is green or not
        if(((closing_amt[i+1] - opening_amt[i+1]) > 0) and ((high[i]<high[i+1]) and (high[i]>low[i+1]) and (low[i]<high[i+1]) and (low[i]>low[i+1]))       and main_account>0):
            #Make a purchase as next candle is green and overlaping also and also wallet have sufficient funds
            risk_account=risk_account_size_update(main_account,risk_account)
            # print(risk_account)
            risk_amount=float(risk_account* (0.01))
            # print(risk_amount)
            # stopprofit=(3*(high[i+1]-low[i+1])) + purchased_price
            stoploss_price=low[i+1]
            stoploss_amount=closing_amt[i]-stoploss_price
            stopprofit_amount=(3*stoploss_amount)
            loss_on_1_quantity=closing_amt[i]-stoploss_price
            quantity_to_purchase=risk_amount/loss_on_1_quantity
            main_account=main_account-(quantity_to_purchase*closing_amt[i])  #updating main account balance after making type purchase
            purchase_history.append(quantity_to_purchase*closing_amt[i])
            portfolio_value_history.append(main_account)
            date_purchased=date[i]
            purchased_price=closing_amt[i]
            curr_trades.put((stoploss_amount,[purchased_price,quantity_to_purchase,stoploss_amount,stopprofit_amount,date_purchased,i]))
            current_purchased_quantity+=quantity_to_purchase



# now check if still some trades remained untraded   ---------> confirm it !!!!  ki isko execute krna hai ya as it is rakhna hai in pending trades ko

while(not(curr_trades.empty())):    
    temp_stop_loss_amount,temp_trade=curr_trades.get()
    temp_stop_loss_price=temp_trade[0]-temp_stop_loss_amount
    temp_stop_profit_price=temp_trade[0]+(3*temp_stop_loss_amount)
    if(closing_amt[-1]<=temp_stop_loss_price):
        #stoploss hit hence sell
        # profit_on_one_quantity=closing_amt[i]-purchased_price
        # total_profit=(quantity*profit_on_one_quantity) + total_profit
        date_sell=date[-1]
        net_loss=temp_stop_loss_amount*temp_trade[1]
        amount_credit_to_bank=(temp_trade[0]-temp_stop_loss_amount)*temp_trade[1]
        main_account=main_account+(amount_credit_to_bank)
        portfolio_value_history.append(main_account)
        sell_history.append(amount_credit_to_bank)
        date_purchased=temp_trade[4]
        risk_account=risk_account_size_update(main_account,risk_account)
        days_taken=days_between(date_purchased,date_sell)
        trading_durations.append(days_taken)
        trades.append([date_purchased,date_sell,"quantity : ",temp_trade[1],"loss",net_loss,main_account,((i-temp_trade[-1])+1),"days taken : ",days_taken])
        loss_array.append(net_loss)
        returns.append(-(net_loss/(temp_trade[0]*temp_trade[1]))*100)
        returns_per_day.append((-(net_loss/(temp_trade[0]*temp_trade[1]))*100)/days_taken)
        num_loss_trades=num_loss_trades+1
        candles_for_loss.append((i-temp_trade[-1])+1)
        candles_req.append((i-temp_trade[-1])+1)
        current_purchased_quantity-=temp_trade[1]
        # initial_amt=initial_amt+(quantity*profit_on_one_quantity)
        # trades.append([date_purchased,date_sell,quantity*profit_on_one_quantity,initial_amt])

        
    elif(closing_amt[-1]>=temp_stop_profit_price):
        #stopprofit hit hence sell
        # profit_on_one_quantity=closing_amt[i]-purchased_price
        # total_profit=(quantity*profit_on_one_quantity) + total_profit
        date_sell=date[-1]
        net_gain=temp_stop_profit_amount*temp_trade[1]
        amount_credit_to_bank=(temp_stop_profit_price)*temp_trade[1]
        main_account=main_account+(amount_credit_to_bank)
        portfolio_value_history.append(main_account)
        sell_history.append(amount_credit_to_bank)
        date_purchased=temp_trade[4]
        risk_account=risk_account_size_update(main_account,risk_account)
        days_taken=days_between(date_purchased,date_sell)
        trading_durations.append(days_taken)
        trades.append([date_purchased,date_sell,"quantity : ",temp_trade[1],"gain",net_gain,main_account,((i-temp_trade[-1])+1),"days taken : ",days_taken])
        gain_array.append(net_gain)
        returns.append((net_gain/(temp_trade[0]*temp_trade[1]))*100)
        returns_per_day.append(((net_gain/(temp_trade[0]*temp_trade[1]))*100)/days_taken)
        num_profitable_trades=num_profitable_trades+1
        candles_for_profit.append((i-temp_trade[-1])+1)
        candles_req.append((i-temp_trade[-1])+1)
        current_purchased_quantity-=temp_trade[1]
        # initial_amt=initial_amt+(quantity*profit_on_one_quantity)
        # trades.append([date_purchased,date_sell,quantity*profit_on_one_quantity,initial_amt])
    else:
        date_sell=date[-1]
        net_gain=(closing_amt[-1]-temp_trade[0])*temp_trade[1]
        amount_credit_to_bank=(closing_amt[-1]-temp_trade[0])*temp_trade[1]
        main_account=main_account+(amount_credit_to_bank)
        portfolio_value_history.append(main_account)
        sell_history.append(amount_credit_to_bank)
        date_purchased=temp_trade[4]
        risk_account=risk_account_size_update(main_account,risk_account)
        days_taken=days_between(date_purchased,date_sell)
        trading_durations.append(days_taken)
        trades.append([date_purchased,date_sell,"quantity : ",temp_trade[1],"gain",net_gain,main_account,((i-temp_trade[-1])+1),"days taken : ",days_taken])
        gain_array.append(net_gain)
        returns.append((net_gain/(temp_trade[0]*temp_trade[1]))*100)
        returns_per_day.append(((net_gain/(temp_trade[0]*temp_trade[1]))*100)/days_taken)
        num_profitable_trades=num_profitable_trades+1
        candles_for_profit.append((i-temp_trade[-1])+1)
        candles_req.append((i-temp_trade[-1])+1)
        current_purchased_quantity-=temp_trade[1]
        # initial_amt=inial_amt+(quantity*profit_on_one_quantity)
        # trades.append([date_purchased,date_sell,quantity*profit_on_one_quantity,initial_amt])
    
portfolio_value_including_purchased_stocks.append([date[-1],main_account])

# Calculate monthly returns
monthly_returns= calculate_monthly_returns(portfolio_value_including_purchased_stocks)

# finding common months in the monthly returns data of our stock and of nifty-50 monthly returns
common_months=[]
months_for_nifty=[]
months_for_our_stock=[]
for abcd in monthly_returns:
    months_for_our_stock.append(abcd[0])
for abcd in monthly_returns:
    months_for_nifty.append(abcd[0])

common_months=[item for item in months_for_nifty if item in months_for_our_stock]

start_index_for_nifty=0
start_index_for_our_stock=0
end_index_for_nifty=0
end_index_for_our_stock=0
for fgh in range(0,len(monthly_returns)):
    if(monthly_returns[fgh][0]==common_months[0]):
        start_index_for_our_stock=fgh
    if(monthly_returns[fgh][0]==common_months[-1]):
        end_index_for_our_stock=fgh
for fgh in range(0,len(monthly_returns_of_nifty_50_in_correct_format)):
    if(monthly_returns_of_nifty_50_in_correct_format[fgh][0]==common_months[0]):
        start_index_for_nifty=fgh
    if(monthly_returns_of_nifty_50_in_correct_format[fgh][0]==common_months[-1]):
        end_index_for_nifty=fgh

stock_returns_percentage=[]
benchmark_returns_percentage=[]
for fgh in range(start_index_for_nifty,end_index_for_nifty+1):
    benchmark_returns_percentage.append(float(monthly_returns_of_nifty_50_in_correct_format[fgh][1][:-1]))
for fgh in range(start_index_for_our_stock,end_index_for_our_stock+1):
    stock_returns_percentage.append(float(monthly_returns[fgh][1][:-1]))

# print(benchmark_returns_percentage)
# print(stock_returns_percentage)
risk_free_rate_percentage = [0.5] * len(stock_returns_percentage)  # Example: constant risk-free rate

# Convert percentage returns to decimal
# stock_returns = np.array(stock_returns_percentage) / 100
stock_returns = [item/100 for item in stock_returns_percentage]
# benchmark_returns = np.array(benchmark_returns_percentage) / 100
benchmark_returns = [item/100 for item in benchmark_returns_percentage]
# risk_free_rate = np.array(risk_free_rate_percentage) / 100
risk_free_rate = [item/100 for item in risk_free_rate_percentage]

# Calculate mean returns
mean_stock_return = np.mean(stock_returns)
mean_benchmark_return = np.mean(benchmark_returns)
mean_rf_return = np.mean(risk_free_rate)

# Calculate deviations from the mean
deviation_stock = stock_returns - mean_stock_return
deviation_benchmark = benchmark_returns - mean_benchmark_return

# Calculate covariance and variance
covariance = np.mean(deviation_stock * deviation_benchmark)
variance_benchmark = np.mean(deviation_benchmark ** 2)

# Calculate Beta
beta = covariance / variance_benchmark

# Calculate expected return using CAPM model
expected_return = mean_rf_return + beta * (mean_benchmark_return - mean_rf_return)

# Calculate Alpha
alpha = mean_stock_return - expected_return

maximum_consecutive_wins=0
maximum_consecutive_losses=0
wins_streak_counter=0
loss_streak_counter=0
data={
    "entry":[],
    "exit":[],
    "net-gain-or-loss":[],
    "candles-req":[],
    "quantity":[]
}
# trades.append([date_purchased,date_sell,"quantity : ",temp_trade[1],"loss",main_account])
for i in trades:
    data["entry"].append(i[0])
    data["exit"].append(i[1])
    data["quantity"].append(i[3])
    if(i[4]=="loss"):
        loss_streak_counter+=1
        if(wins_streak_counter>maximum_consecutive_wins):
            maximum_consecutive_wins=wins_streak_counter
        wins_streak_counter=0
        data["net-gain-or-loss"].append(-(i[5]))
    else:
        wins_streak_counter+=1
        if(loss_streak_counter>maximum_consecutive_losses):
            maximum_consecutive_losses=loss_streak_counter
        loss_streak_counter=0
        data["net-gain-or-loss"].append(i[5])
    data["candles-req"].append(i[-1])

new_data=pd.DataFrame(data)
new_data.to_csv('output.csv')
print(trades)
total_num_trades=num_loss_trades+num_profitable_trades

print("Amount Invested : ",input_account)
# print(portfolio_value_including_purchased_stocks)
print("Total amount now in my pocket: ",main_account)
print("1.) Total number of trades : ",total_num_trades)
print("2.) No of trades in loss : ",num_loss_trades)
print("3.) No of trades in profit : ",num_profitable_trades)
win_rate=(num_profitable_trades/total_num_trades)*100
print("4.) Win Rate : ",win_rate)
print("5.) Average Gain per Winning Trade : ", (sum(gain_array)/len(gain_array)))
print("6.) Average Loss per Losing Trade : ", (sum(loss_array)/len(loss_array)))
print("7.) Profit Factor : ", (sum(gain_array)/sum(loss_array)))
win_rate=(num_profitable_trades/total_num_trades)*100
average_gain=(sum(gain_array)/len(gain_array))
average_loss=(sum(loss_array)/len(loss_array))
print("8.) Expected Value : ", (((win_rate/100)*average_gain)-((1-(win_rate/100))*average_loss)))

running_maximum=closing_amt[0]
drawdowns=[]
for s in closing_amt:
    if(s>running_maximum):
        running_maximum=s
    curr_drawdown=((s-running_maximum)/running_maximum)*100
    drawdowns.append(curr_drawdown)
print("9.) Maximum Drawdown : ",min(drawdowns))
mean_return=sum(returns_per_day)/len(returns_per_day)
squared_sum_for_standard_deviation=0
squared_sum_for_downward_deviation=0
downwards_returns_count=0
for x in returns_per_day:
    squared_sum_for_standard_deviation=squared_sum_for_standard_deviation+((x-mean_return)**2)
    if(x<mean_return):
        # we have taken the mean return as our threshhold value
        squared_sum_for_downward_deviation=squared_sum_for_downward_deviation+((x-mean_return)**2)
        downwards_returns_count+=1

variance_for_standard_deviation=squared_sum_for_standard_deviation/(len(returns_per_day)-1)
variance_for_downward_deviation=squared_sum_for_downward_deviation/downwards_returns_count
standard_deviation=(variance_for_standard_deviation)**0.5
downward_deviation=(variance_for_downward_deviation)**0.5
risk_free_return_per_day=6/365
sharpe_ratio=(mean_return-risk_free_return_per_day)/standard_deviation
sortino_ratio=(mean_return-risk_free_return_per_day)/downward_deviation
print("10.) Mean Return : ",mean_return)
print("11.) Standard deviation : ",standard_deviation)
print("11.) Sharpe Ratio : ",sharpe_ratio)
print("12.) Sortino Ratio : ",sortino_ratio)
print("13.) Return on investment (ROI) : ",((main_account-input_account)/input_account)*100)
cumulative_return=1
for r in returns:
    cumulative_return=cumulative_return * ((r/100)+1)
cumulative_return-=1
cumulative_return*=100
print("14.) Cumulative return : ",cumulative_return)
total_duration_in_days=days_between(date[0],date[-1])
total_duration_in_years=days_between(date[0],date[-1])/365
annualized_return=(((1+(cumulative_return/100))**(1/total_duration_in_years))-1)*100
print("15.) Annualized return : ",annualized_return)
print("16.) Calmar Ratio : ",(annualized_return/(abs(min(drawdowns)))))
print("17.) Volatility/Annualized standard deviation : ",(standard_deviation*((252)**0.5)))  #assuming 252 trading 
print("18.) Alpha : ",alpha)  #The measure of the strategy's performance on a risk-adjusted basis relative to a 
print("19.) Beta : ",beta)  #The measure of the strategy's sensitivity to movements in the benchmark index.
print("20.) Maximum consecutive Wins : ",maximum_consecutive_wins)  #The longest streak of consecutive winning trades.
print("21.) Maximum consecutive Losses : ",maximum_consecutive_losses) #The longest streak of consecutive losing trades.
print("22.) Recovery Factor : ",(cumulative_return/(abs(min(drawdowns))))) #The ratio of net profit to the maximum drawdown. It indicates how quickly the strategy recovers from drawdowns.
print("23.) Payoff Ratio : ",((sum(gain_array)/len(gain_array))/(sum(loss_array)/len(loss_array))))  #The ratio of average profit per trade to the average loss per trade.
print("24.) Profitability Index : ",(sum(gain_array)/sum(loss_array))) #The sum of all profits divided by the sum of all losses.
print("25.) Trade Duration : ",(sum(trading_durations)/len(trading_durations))) #The average time a trade is held.
average_portfolio_value=(sum(portfolio_value_history)/len(portfolio_value_history))
total_purchase_value=sum(purchase_history)
total_sell_value=sum(sell_history)
print("26.) Turnover Rate for complete period : ",((min(total_purchase_value,total_sell_value))/average_portfolio_value)) #How frequently assets are bought and sold within the strategy.
print("26.) Turnover Rate for one year : ",(((min(total_purchase_value,total_sell_value))/average_portfolio_value)/total_duration_in_years))
print("Net Gain or loss month wise : ",months_wise_gain_or_loss)
print("Profit : ",candles_for_profit) 
print("Loss : ",candles_for_loss)
