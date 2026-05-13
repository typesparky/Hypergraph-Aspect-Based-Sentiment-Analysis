import re
import requests
import json

def sync_kanban():
    file_path = "/Users/robertalexandrou/Documents/Econometrics Courses/Thesis/kanban/ABSC-Hypergraph-Thesis.md"
    url = "http://localhost:8642/api/kanban/tasks" # API endpoint for Hermes Kanban
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Simple regex to find tasks like "- [ ] Task name" or "- [x] Task name"
    tasks = re.findall(r'- \[(.)\] (.*)', content)
    
    for status_char, task_name in tasks:
        task_data = {
            "content": task_name,
            "status": "completed" if status_char == 'x' else "pending",
            "board": "ABSC-Hypergraph-Thesis"
        }
        # In a real setup, we would POST this to the API. 
        # Since we are using the file-based Kanban, I'll print the intended sync actions.
        print(f"Syncing to Dashboard: {task_name} | Status: {task_data['status']}")

if __name__ == "__main__":
    print("Kanban Sync Simulation initiated...")
    sync_kanban()
