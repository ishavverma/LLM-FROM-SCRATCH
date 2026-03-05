import json
import os
import argparse
import sys

class HITLReviewer:
    """Human-in-the-Loop reviewer for dataset preparation."""
    
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.chunks = []
        self.approved_chunks = []
        
    def load_chunks(self):
        if not os.path.exists(self.input_file):
            print(f"Error: Internal dataset {self.input_file} not found.")
            sys.exit(1)
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
            
    def save_approved(self):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.approved_chunks, f, indent=4)
            
    def start_review(self):
        self.load_chunks()
        print(f"\n--- Starting HITL Review ({len(self.chunks)} chunks) ---")
        print("Commands: (a)pprove, (r)eject, (q)uit")
        
        for i, chunk_data in enumerate(self.chunks):
            if isinstance(chunk_data, str):
                chunk_text = chunk_data
                chunk_type = "base"
            else:
                chunk_text = chunk_data.get("text", "")
                chunk_type = chunk_data.get("type", "base")
                
            print(f"\nChunk {i+1}/{len(self.chunks)} (Type: {chunk_type}):")
            print("-" * 40)
            # Display preview
            print(chunk_text[:500] + ("..." if len(chunk_text) > 500 else ""))
            print("-" * 40)
            
            while True:
                choice = input("Action (a/r/q): ").strip().lower()
                if choice == 'a':
                    self.approved_chunks.append({"text": chunk_text, "type": chunk_type})
                    print("Approved.")
                    break
                elif choice == 'r':
                    print("Rejected.")
                    break
                elif choice == 'q':
                    print("Saving and quitting...")
                    self.save_approved()
                    return
                else:
                    print("Invalid choice. Please enter 'a', 'r', or 'q'.")
                    
        self.save_approved()
        print("\nReview complete. Approved dataset saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human-in-the-Loop Review CLI")
    parser.add_argument("--input", required=True, help="Path to unreviewed chunks JSON")
    parser.add_argument("--output", required=True, help="Path to save approved chunks JSON")
    args = parser.parse_args()
    
    reviewer = HITLReviewer(args.input, args.output)
    reviewer.start_review()
