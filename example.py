from main import run_analysis

code1 = """
def process_sales_data(transactions):
    total_revenue = 0
    total_quantity = 0
    regions = {}
    
    for txn in transactions:
        discounted_amount = txn['amount'] * (1 - txn['discount'])
        total_revenue += discounted_amount
        total_quantity += txn['quantity']
        
        region = txn['region']
        if region not in regions:
            regions[region] = {'revenue': 0, 'count': 0}
        regions[region]['revenue'] += discounted_amount
        regions[region]['count'] += 1
    
    return {
        'total_revenue': round(total_revenue, 2),
        'total_quantity': total_quantity,
        'avg_transaction': round(total_revenue / len(transactions), 2) if transactions else 0,
        'regions': regions
    }
"""

code2 = """
def process_sales_data(transactions):
    if not transactions:
        return {
            'total_revenue': 0,
            'total_quantity': 0,
            'avg_transaction': 0,
            'regions': {}
        }
    
    total_revenue = sum([
        txn['amount'] * (1 - txn.get('discount', 0))
        for txn in transactions
    ])
    
    total_quantity = sum([txn['quantity'] for txn in transactions])
    
    regions = {}
    for txn in transactions:
        region = txn['region']
        discounted = txn['amount'] * (1 - txn.get('discount', 0))
        
        if region in regions:
            regions[region]['revenue'] += discounted
            regions[region]['count'] += 1
        else:
            regions[region] = {'revenue': discounted, 'count': 1}
    
    return {
        'total_revenue': round(total_revenue, 2),
        'total_quantity': total_quantity,
        'avg_transaction': round(total_revenue / len(transactions), 2),
        'regions': regions
    }
"""

result = run_analysis(code1, code2, max_iterations=5, verbose=3)
