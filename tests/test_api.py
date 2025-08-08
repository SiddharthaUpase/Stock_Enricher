#!/usr/bin/env python3
"""
Simple test script for the Flask API
"""

import requests
import time
import json

API_BASE = "http://localhost:8080"

def test_api():
    """Test the Flask API endpoints"""
    
    print("🧪 Testing Stock Enrichment API")
    print("=" * 40)
    
    # 1. Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return
    
    # 2. Upload CSV file
    print("\n2. Testing CSV upload...")
    try:
        # Generate dummy data first
        import sys
        sys.path.append('..')
        from tests.generate_dummy_data import generate_dummy_portfolio_data, save_to_csv
        
        # Generate and save dummy data
        portfolio_data = generate_dummy_portfolio_data()
        save_to_csv(portfolio_data, 'temp_test_portfolio.csv')
        
        with open('temp_test_portfolio.csv', 'rb') as f:
            files = {'file': ('dummy_portfolio.csv', f, 'text/csv')}
            response = requests.post(f"{API_BASE}/upload", files=files)
        
        print(f"   Status: {response.status_code}")
        upload_result = response.json()
        print(f"   Response: {upload_result}")
        
        if response.status_code != 200:
            print("   ❌ Upload failed")
            return
        
        task_id = upload_result['task_id']
        print(f"   ✅ Task started: {task_id}")
        
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return
    
    # 3. Poll status
    print("\n3. Polling task status...")
    max_polls = 60  # Max 5 minutes
    poll_count = 0
    
    while poll_count < max_polls:
        try:
            response = requests.get(f"{API_BASE}/status/{task_id}")
            status_result = response.json()
            
            status = status_result.get('status')
            progress = status_result.get('progress', {})
            message = status_result.get('message', '')
            
            percentage = progress.get('percentage', 0)
            current_company = progress.get('current_company', '')
            
            print(f"   Poll #{poll_count + 1}: {status} - {percentage:.1f}% - {message}")
            if current_company:
                print(f"     Processing: {current_company}")
            
            if status == 'completed':
                print("   ✅ Task completed successfully!")
                break
            elif status == 'failed':
                error = status_result.get('error', 'Unknown error')
                print(f"   ❌ Task failed: {error}")
                return
            
            time.sleep(5)  # Wait 5 seconds
            poll_count += 1
            
        except Exception as e:
            print(f"   ❌ Status check failed: {e}")
            return
    
    if poll_count >= max_polls:
        print("   ⏰ Timeout waiting for task completion")
        return
    
    # 4. Download result
    print("\n4. Testing download...")
    try:
        response = requests.get(f"{API_BASE}/download/{task_id}")
        
        if response.status_code == 200:
            # Save the downloaded file
            output_filename = f"test_enriched_result_{task_id[:8]}.csv"
            with open(output_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"   ✅ Downloaded result to: {output_filename}")
            
            # Show first few lines
            print("   📄 First few lines of result:")
            lines = response.content.decode('utf-8').split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"     {line}")
            
        else:
            print(f"   ❌ Download failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
    
    # 5. List tasks
    print("\n5. Listing all tasks...")
    try:
        response = requests.get(f"{API_BASE}/tasks")
        tasks_result = response.json()
        
        print(f"   Status: {response.status_code}")
        print(f"   Tasks: {json.dumps(tasks_result, indent=2)}")
        
    except Exception as e:
        print(f"   ❌ Task listing failed: {e}")
    
    print("\n🎉 API test completed!")
    
    # Cleanup
    try:
        import os
        if os.path.exists('temp_test_portfolio.csv'):
            os.remove('temp_test_portfolio.csv')
        print("   🧹 Cleaned up temporary files")
    except:
        pass

if __name__ == "__main__":
    print("Make sure the Flask API is running on http://localhost:8080")
    print("Run: python app.py")
    print()
    input("Press Enter when ready to test...")
    
    test_api() 