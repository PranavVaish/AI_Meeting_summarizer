import requests
import os
# Your API key
api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Define headers with authorization
headers = {'authorization': api_key}

try:
    # Make request to the account endpoint
    print("Testing AssemblyAI API key...")
    response = requests.get('https://api.assemblyai.com/v2/account', headers=headers)
    
    # Check status code
    print(f"Status code: {response.status_code}")
    
    # Interpret the result
    if response.status_code == 200:
        account_data = response.json()
        print("\nAPI KEY IS WORKING! ✅")
        print("\nAccount information:")
        print(f"  - Status: {account_data.get('status', 'N/A')}")
        print(f"  - Balance: {account_data.get('balance', 'N/A')}")
        print(f"  - Pending: {account_data.get('pending', 'N/A')}")
        
        # Show additional account details if available
        if 'current_plan' in account_data:
            print(f"  - Current plan: {account_data['current_plan']}")
    
    elif response.status_code == 401:
        print("\nAPI KEY IS NOT VALID! ❌")
        print("You received an 'Unauthorized' response. Please check your API key and try again.")
    
    else:
        print(f"\nUnexpected response (HTTP {response.status_code})")
        print("Response content:", response.text)

except requests.exceptions.RequestException as e:
    print(f"\nError making the request: {e}")
    print("Please check your internet connection and try again.")