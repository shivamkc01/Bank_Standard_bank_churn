import pandas as pd
recommendation_data = pd.DataFrame({
    'Churn_Type': ['Young Customers', 'High-Income Customers', 'Female Customers', 'Recently Married Customers', 
                   'Long-Term Customers', 'Low Transaction Activity Customers', 'High Transaction Activity Customers'],
    'Recommendation': [
        'Offer student or youth banking accounts with no monthly fees and mobile banking features.', 
        'Recommend premium banking services with personalized wealth management options and exclusive rewards.', 
        'Provide specialized financial planning services tailored to women\'s financial needs, such as investment advice or savings plans.', 
        'Offer joint accounts or mortgage services with competitive rates for home loans.', 
        'Provide loyalty rewards or incentives for long-term customers, such as cashback on transactions or reduced fees.', 
        'Recommend budgeting tools or financial education resources to help them manage their finances better.', 
        'Offer premium credit cards with rewards points, travel benefits, or cashback incentives.'
    ]
})

recommendation_data.to_csv("../data/dummy_recommendation_dataset/recommendation_dataset.csv", index=False)
