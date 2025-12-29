"""
Comment Filtering UI
A GUI that reads JSONL items from a file and shows comments one by one.
Allows the user to filter them by clicking "Keep" or "Discard" buttons.
The filtered comments are saved to a new JSONL file.
Supports caching to resume from previous runs.
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter import font as tkfont
from pathlib import Path
import hashlib


class CommentFilteringApp:
    # Font settings - adjust these to change UI font sizes
    FONT_SIZE_NORMAL = 14
    FONT_SIZE_LARGE = 16
    FONT_SIZE_HEADER = 18
    
    def __init__(self, root):
        self.root = root
        self.root.title("Comment Filtering UI")
        self.root.geometry("1000x800")
        
        # Data storage
        self.all_entities = []  # List of (paper_id, comment, response) tuples
        self.current_index = 0
        self.kept_entities = []
        self.discarded_entities = []
        self.input_file = None
        self.cache_file = None
        
        # Configure fonts
        self.setup_fonts()
        self.setup_ui()
        
    def setup_fonts(self):
        """Configure larger fonts for better readability"""
        self.default_font = tkfont.nametofont("TkDefaultFont")
        self.default_font.configure(size=self.FONT_SIZE_NORMAL)
        
        self.text_font = tkfont.Font(family="TkDefaultFont", size=self.FONT_SIZE_LARGE)
        self.header_font = tkfont.Font(
            family="TkDefaultFont", 
            size=self.FONT_SIZE_HEADER, 
            weight="bold"
        )
        self.button_font = tkfont.Font(family="TkDefaultFont", size=self.FONT_SIZE_NORMAL)
        
        # Configure ttk styles
        style = ttk.Style()
        style.configure("TLabel", font=("TkDefaultFont", self.FONT_SIZE_NORMAL))
        style.configure("TButton", font=("TkDefaultFont", self.FONT_SIZE_NORMAL))
        style.configure(
            "TLabelframe.Label", 
            font=("TkDefaultFont", self.FONT_SIZE_NORMAL, "bold")
        )
        
    def get_cache_path(self, input_file):
        """Generate a cache file path based on the input file"""
        input_path = Path(input_file)
        # Create a hash of the file path to make unique cache files
        file_hash = hashlib.md5(str(input_path.absolute()).encode()).hexdigest()[:8]
        cache_name = f".{input_path.stem}_{file_hash}_cache.json"
        return input_path.parent / cache_name
        
    def load_cache(self):
        """Load cached progress from previous session"""
        if not self.cache_file or not self.cache_file.exists():
            return False
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Verify the cache matches our current data
            if cache_data.get('total_entities') != len(self.all_entities):
                return False
                
            # Restore status for each entity
            cached_statuses = cache_data.get('statuses', {})
            for idx, entity in enumerate(self.all_entities):
                key = f"{entity['paper_id']}_{idx}"
                if key in cached_statuses:
                    entity['status'] = cached_statuses[key]
                    
            # Restore current index
            self.current_index = cache_data.get('current_index', 0)
            if self.current_index >= len(self.all_entities):
                self.current_index = 0
                
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
            
    def save_cache(self):
        """Save current progress to cache file"""
        if not self.cache_file or not self.all_entities:
            return
            
        try:
            # Build status dictionary
            statuses = {}
            for idx, entity in enumerate(self.all_entities):
                key = f"{entity['paper_id']}_{idx}"
                if entity.get('status'):
                    statuses[key] = entity['status']
                    
            cache_data = {
                'total_entities': len(self.all_entities),
                'current_index': self.current_index,
                'statuses': statuses,
                'input_file': str(self.input_file)
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
            
    def clear_cache(self):
        """Delete the cache file"""
        if self.cache_file and self.cache_file.exists():
            try:
                self.cache_file.unlink()
            except Exception as e:
                print(f"Error clearing cache: {e}")
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(file_frame, text="Open JSONL File", command=self.open_file).pack(side=tk.RIGHT)
        
        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="Progress: 0/0")
        self.progress_label.pack(side=tk.LEFT)
        
        self.stats_label = ttk.Label(progress_frame, text="Kept: 0 | Maybe: 0 | Discarded: 0")
        self.stats_label.pack(side=tk.RIGHT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Paper ID frame
        paper_frame = ttk.LabelFrame(main_frame, text="Paper ID", padding="5")
        paper_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.paper_id_label = ttk.Label(
            paper_frame, 
            text="", 
            font=self.header_font
        )
        self.paper_id_label.pack(fill=tk.X)
        
        # Comment frame
        comment_frame = ttk.LabelFrame(
            main_frame, 
            text="Comment (Reviewer)", 
            padding="5"
        )
        comment_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.comment_text = scrolledtext.ScrolledText(
            comment_frame, 
            wrap=tk.WORD, 
            height=8,
            font=self.text_font
        )
        self.comment_text.pack(fill=tk.BOTH, expand=True)
        self.comment_text.config(state=tk.DISABLED)
        
        # Response frame
        response_frame = ttk.LabelFrame(
            main_frame, 
            text="Response (Authors)", 
            padding="5"
        )
        response_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.response_text = scrolledtext.ScrolledText(
            response_frame, 
            wrap=tk.WORD, 
            height=8,
            font=self.text_font
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)
        self.response_text.config(state=tk.DISABLED)
        
        # Navigation and action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Navigation buttons on the left
        nav_frame = ttk.Frame(button_frame)
        nav_frame.pack(side=tk.LEFT)
        
        self.prev_btn = ttk.Button(
            nav_frame, 
            text="← Previous", 
            command=self.previous_entity
        )
        self.prev_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.next_btn = ttk.Button(
            nav_frame, 
            text="Next →", 
            command=self.next_entity
        )
        self.next_btn.pack(side=tk.LEFT)
        
        # Action buttons in the center
        action_frame = ttk.Frame(button_frame)
        action_frame.pack(side=tk.LEFT, expand=True)
        
        self.discard_btn = ttk.Button(
            action_frame, 
            text="✗ Discard (D)", 
            command=self.discard_entity
        )
        self.discard_btn.pack(side=tk.LEFT, padx=10)
        
        self.maybe_btn = ttk.Button(
            action_frame, 
            text="? Maybe (M)", 
            command=self.maybe_entity
        )
        self.maybe_btn.pack(side=tk.LEFT, padx=10)
        
        self.keep_btn = ttk.Button(
            action_frame, 
            text="✓ Keep (K)", 
            command=self.keep_entity
        )
        self.keep_btn.pack(side=tk.LEFT, padx=10)
        
        # Save button on the right
        self.save_btn = ttk.Button(
            button_frame, 
            text="Save Results", 
            command=self.save_results
        )
        self.save_btn.pack(side=tk.RIGHT)
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.previous_entity())
        self.root.bind('<Right>', lambda e: self.next_entity())
        self.root.bind('<k>', lambda e: self.keep_entity())
        self.root.bind('<K>', lambda e: self.keep_entity())
        self.root.bind('<m>', lambda e: self.maybe_entity())
        self.root.bind('<M>', lambda e: self.maybe_entity())
        self.root.bind('<d>', lambda e: self.discard_entity())
        self.root.bind('<D>', lambda e: self.discard_entity())
        
        # Initially disable buttons
        self.update_button_states()
        
    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Select JSONL File",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.input_file = file_path
        self.file_label.config(text=Path(file_path).name)
        
        # Set up cache file
        self.cache_file = self.get_cache_path(file_path)
        
        # Load and parse the file
        self.all_entities = []
        self.kept_entities = []
        self.discarded_entities = []
        self.current_index = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        paper_id = data.get('paper_id', 'Unknown')
                        response = data.get('response', {})
                        
                        # Handle case where response might be a string or dict
                        if isinstance(response, str):
                            try:
                                response = json.loads(response)
                            except json.JSONDecodeError:
                                continue
                        
                        entities = response.get('entities', [])
                        
                        for entity in entities:
                            comment = entity.get('comment', '')
                            resp = entity.get('response', '')
                            if comment or resp:  # Only add if there's content
                                self.all_entities.append({
                                    'paper_id': paper_id,
                                    'comment': comment,
                                    'response': resp,
                                    'original_data': data,  # Store all original fields
                                    'status': None  # None = not reviewed, 'keep', 'discard'
                                })
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue
            
            if self.all_entities:
                # Try to load cached progress
                cache_loaded = self.load_cache()
                if cache_loaded:
                    reviewed = sum(
                        1 for e in self.all_entities if e.get('status')
                    )
                    messagebox.showinfo(
                        "Resumed from Cache", 
                        f"Loaded {len(self.all_entities)} comment/response pairs.\n"
                        f"Resumed progress: {reviewed} items already reviewed.\n"
                        f"Continuing from item {self.current_index + 1}."
                    )
                else:
                    messagebox.showinfo(
                        "File Loaded", 
                        f"Loaded {len(self.all_entities)} comment/response pairs"
                    )
                self.display_current_entity()
            else:
                messagebox.showwarning("No Data", "No valid entities found in the file")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            
        self.update_button_states()
        self.update_progress()
        
    def display_current_entity(self):
        if not self.all_entities or self.current_index >= len(self.all_entities):
            return
            
        entity = self.all_entities[self.current_index]
        
        # Update paper ID
        self.paper_id_label.config(text=f"Paper: {entity['paper_id']}")
        
        # Update comment text
        self.comment_text.config(state=tk.NORMAL)
        self.comment_text.delete('1.0', tk.END)
        self.comment_text.insert(tk.END, entity['comment'])
        self.comment_text.config(state=tk.DISABLED)
        
        # Update response text
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete('1.0', tk.END)
        self.response_text.insert(tk.END, entity['response'])
        self.response_text.config(state=tk.DISABLED)
        
        # Update status indicator in title
        status = entity.get('status')
        if status == 'keep':
            status_text = " [KEPT]"
        elif status == 'maybe':
            status_text = " [MAYBE]"
        elif status == 'discard':
            status_text = " [DISCARDED]"
        else:
            status_text = ""
        self.root.title(f"Comment Filtering UI - Item {self.current_index + 1}/{len(self.all_entities)}{status_text}")
        
    def update_progress(self):
        total = len(self.all_entities)
        kept = sum(1 for e in self.all_entities if e.get('status') == 'keep')
        maybe = sum(1 for e in self.all_entities if e.get('status') == 'maybe')
        discarded = sum(1 for e in self.all_entities if e.get('status') == 'discard')
        reviewed = kept + maybe + discarded
        
        self.progress_label.config(text=f"Progress: {self.current_index + 1}/{total} | Reviewed: {reviewed}/{total}")
        self.stats_label.config(text=f"Kept: {kept} | Maybe: {maybe} | Discarded: {discarded}")
        
        if total > 0:
            self.progress_bar['value'] = (reviewed / total) * 100
        
    def update_button_states(self):
        has_data = len(self.all_entities) > 0
        
        self.prev_btn.config(state=tk.NORMAL if has_data and self.current_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if has_data and self.current_index < len(self.all_entities) - 1 else tk.DISABLED)
        self.keep_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.maybe_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.discard_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        
    def keep_entity(self):
        if not self.all_entities:
            return
        self.all_entities[self.current_index]['status'] = 'keep'
        self.save_cache()  # Auto-save progress
        self.update_progress()
        self.auto_advance()
        
    def maybe_entity(self):
        if not self.all_entities:
            return
        self.all_entities[self.current_index]['status'] = 'maybe'
        self.save_cache()  # Auto-save progress
        self.update_progress()
        self.auto_advance()
        
    def discard_entity(self):
        if not self.all_entities:
            return
        self.all_entities[self.current_index]['status'] = 'discard'
        self.save_cache()  # Auto-save progress
        self.update_progress()
        self.auto_advance()
        
    def auto_advance(self):
        """Automatically advance to the next unreviewed entity"""
        if self.current_index < len(self.all_entities) - 1:
            self.current_index += 1
            self.display_current_entity()
            self.update_button_states()
            self.save_cache()  # Save position
        else:
            self.display_current_entity()  # Refresh to show status
            
    def next_entity(self):
        if self.current_index < len(self.all_entities) - 1:
            self.current_index += 1
            self.display_current_entity()
            self.update_button_states()
            self.save_cache()  # Save position
            
    def previous_entity(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_entity()
            self.update_button_states()
            self.save_cache()  # Save position
            
    def save_results(self):
        if not self.all_entities:
            messagebox.showwarning("No Data", "No data to save")
            return
            
        # Get save location
        default_name = "filtered_comments.jsonl"
        if self.input_file:
            input_path = Path(self.input_file)
            default_name = f"{input_path.stem}_filtered.jsonl"
            
        file_path = filedialog.asksaveasfilename(
            title="Save Filtered Results",
            defaultextension=".jsonl",
            initialfile=default_name,
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Save kept entities
            kept_entities = [e for e in self.all_entities if e.get('status') == 'keep']
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for entity in kept_entities:
                    # Start with original data and add/update with current entity info
                    output = entity.get('original_data', {}).copy()
                    output['comment'] = entity['comment']
                    output['response'] = entity['response']
                    # Remove the entities array since we're outputting individual items
                    output.pop('response', None)  # Remove original response dict
                    output['filtered_comment'] = entity['comment']
                    output['filtered_response'] = entity['response']
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
                    
            # Save maybe entities to a separate file
            maybe_path = Path(file_path).with_stem(Path(file_path).stem + "_maybe")
            maybe_entities = [e for e in self.all_entities if e.get('status') == 'maybe']
            
            with open(maybe_path, 'w', encoding='utf-8') as f:
                for entity in maybe_entities:
                    # Start with original data and add/update with current entity info
                    output = entity.get('original_data', {}).copy()
                    output.pop('response', None)  # Remove original response dict
                    output['filtered_comment'] = entity['comment']
                    output['filtered_response'] = entity['response']
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
                    
            # Also save discarded entities to a separate file
            discarded_path = Path(file_path).with_stem(Path(file_path).stem + "_discarded")
            discarded_entities = [e for e in self.all_entities if e.get('status') == 'discard']
            
            with open(discarded_path, 'w', encoding='utf-8') as f:
                for entity in discarded_entities:
                    # Start with original data and add/update with current entity info
                    output = entity.get('original_data', {}).copy()
                    output.pop('response', None)  # Remove original response dict
                    output['filtered_comment'] = entity['comment']
                    output['filtered_response'] = entity['response']
                    f.write(json.dumps(output, ensure_ascii=False) + '\n')
                    
            messagebox.showinfo(
                "Saved", 
                f"Saved {len(kept_entities)} kept items to:\n{file_path}\n\n"
                f"Saved {len(maybe_entities)} maybe items to:\n{maybe_path}\n\n"
                f"Saved {len(discarded_entities)} discarded items to:\n{discarded_path}"
            )
            
            # Ask if user wants to clear the cache
            if self.cache_file and self.cache_file.exists():
                if messagebox.askyesno(
                    "Clear Cache?",
                    "Results saved successfully.\n\n"
                    "Do you want to clear the progress cache?\n"
                    "(Select 'No' if you may want to continue editing later)"
                ):
                    self.clear_cache()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")


def main():
    root = tk.Tk()
    app = CommentFilteringApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()